"""
This file is ported from AllenNLP original implementation. (Under Apache 2.0 license)
https://github.com/allenai/allennlp/blob/main/allennlp/commands/train.py
"""

import os
from os import PathLike
from typing import Any
from typing import cast
from typing import List
from typing import Optional
from typing import Union

from optuna import logging
from optuna import Trial
from optuna import TrialPruned
from optuna._imports import try_import
from optuna.storages import _CachedStorage
from optuna.storages import InMemoryStorage


with try_import() as _allennlp_imports:
    from allennlp.commands.train import TrainModel
    from allennlp.common import logging as common_logging
    from allennlp.common import Params
    from allennlp.common import util as common_util
    from allennlp.common.checks import check_for_gpu
    from allennlp.common.checks import ConfigurationError
    from allennlp.common.plugins import import_plugins
    from allennlp.data import Vocabulary
    from allennlp.models.archival import archive_model
    from allennlp.models.archival import CONFIG_NAME
    from allennlp.models.archival import verify_include_in_archive
    from allennlp.models.model import _DEFAULT_WEIGHTS
    from allennlp.models.model import Model
    from allennlp.training import util as training_util
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

if not _allennlp_imports.is_successful():
    TrainModel = None  # type: ignore  # NOQA
    common_logging = None  # type: ignore  # NOQA
    Params = None  # type: ignore  # NOQA
    common_util = None  # type: ignore  # NOQA
    check_for_gpu = None  # type: ignore  # NOQA
    ConfigurationError = None  # type: ignore  # NOQA
    import_plugins = None  # type: ignore  # NOQA
    Vocabulary = None  # type: ignore  # NOQA
    archive_model = None  # type: ignore  # NOQA
    CONFIG_NAME = None  # type: ignore  # NOQA
    verify_include_in_archive = None  # type: ignore  # NOQA
    _DEFAULT_WEIGHTS = None  # type: ignore  # NOQA
    Model = None  # type: ignore  # NOQA
    training_util = None  # type: ignore  # NOQA
    torch = None  # type: ignore  # NOQA
    dist = None  # type: ignore  # NOQA
    mp = None  # type: ignore  # NOQA


with try_import() as _allennlp_meta_imports:
    from allennlp.common.meta import Meta
    from allennlp.common.meta import META_NAME

if not _allennlp_meta_imports.is_successful():
    Meta = None  # type: ignore  # NOQA
    META_NAME = None  # type: ignore  # NOQA


logger = logging.get_logger(__name__)


def _train_model_with_optuna(
    params: Params,
    serialization_dir: Union[str, PathLike],
    recover: bool = False,
    force: bool = False,
    node_rank: int = 0,
    include_package: List[str] = None,
    dry_run: bool = False,
    file_friendly_logging: bool = False,
    trial: Trial = None,
) -> Optional[Model]:
    """Extended AllenNLP training utility.

    Args:
        params:
            A parameter object specifying an AllenNLP Experiment.
        serialization_dir : `str`
            The directory in which to save results and logs.
        recover : `bool`, optional (default=`False`)
            If `True`, we will try to recover a training run from an existing serialization
            directory.  This is only intended for use when something actually crashed during the
            middle of a run.  For continuing training a model on new data,
            see `Model.from_archive`.
        force : `bool`, optional (default=`False`)
            If `True`, we will overwrite the serialization directory if it already exists.
        node_rank : `int`, optional
            Rank of the current node in distributed training
        include_package : `List[str]`, optional
            In distributed mode, extra packages mentioned will be imported in trainer workers.
        dry_run : `bool`, optional (default=`False`)
            Do not train a model, but create a vocabulary, show dataset statistics and other
            training information.
        file_friendly_logging : `bool`, optional (default=`False`)
            If `True`, we add newlines to tqdm output, even on an interactive terminal, and we slow
            down tqdm's output to only once every 10 seconds.
    """

    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    training_util.create_serialization_dir(params, serialization_dir, recover, force)
    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    if Meta is not None:
        meta = Meta.new()
        meta.to_file(os.path.join(serialization_dir, META_NAME))

    include_in_archive = params.pop("include_in_archive", None)
    verify_include_in_archive(include_in_archive)

    distributed_params = params.params.pop("distributed", None)
    # If distributed isn't in the config and the config contains strictly
    # one cuda device, we just run a single training process.
    if distributed_params is None:
        model = _train_worker_with_optuna(
            process_rank=0,
            params=params,
            serialization_dir=serialization_dir,
            include_package=include_package,
            dry_run=dry_run,
            file_friendly_logging=file_friendly_logging,
            trial=trial,
        )

        if not dry_run:
            archive_model(serialization_dir, include_in_archive=include_in_archive)
        return model

    # Otherwise, we are running multiple processes for training.
    else:
        common_logging.prepare_global_logging(
            serialization_dir,
            rank=0,
            world_size=1,
        )

        # We are careful here so that we can raise a good error if someone
        # passed the wrong thing - cuda_devices are required.
        device_ids = distributed_params.pop("cuda_devices", None)
        multi_device = isinstance(device_ids, list) and len(device_ids) > 1
        num_nodes = distributed_params.pop("num_nodes", 1)

        if not (multi_device or num_nodes > 1):
            raise ConfigurationError(
                "Multiple cuda devices/nodes need to be configured to run distributed training."
            )
        check_for_gpu(device_ids)

        primary_addr = distributed_params.pop("primary_address", "127.0.0.1")
        if primary_addr in ("127.0.0.1", "0.0.0.0", "localhost"):
            # If running locally, we can automatically find an open port if one is not specified.
            primary_port = (
                distributed_params.pop("primary_port", None) or common_util.find_open_port()
            )
        else:
            # Otherwise we require that the port be specified.
            primary_port = distributed_params.pop("primary_port")

        num_procs = len(device_ids)
        world_size = num_nodes * num_procs

        # Creating `Vocabulary` objects from workers could be problematic since
        # the data loaders in each worker will yield only `rank` specific
        # instances. Hence it is safe to construct the vocabulary and write it
        # to disk before initializing the distributed context. The workers will
        # load the vocabulary from the path specified.
        vocab_dir = os.path.join(serialization_dir, "vocabulary")
        if recover:
            vocab = Vocabulary.from_files(vocab_dir)
        else:
            vocab = training_util.make_vocab_from_params(
                params.duplicate(), serialization_dir, print_statistics=dry_run
            )
        params["vocabulary"] = {
            "type": "from_files",
            "directory": vocab_dir,
            "padding_token": vocab._padding_token,
            "oov_token": vocab._oov_token,
        }

        logger.info(
            "Switching to distributed training mode since multiple GPUs are configured | "
            f"Primary is at: {primary_addr}:{primary_port} | Rank of this node: {node_rank} | "
            f"Number of workers in this node: {num_procs} | Number of nodes: {num_nodes} | "
            f"World size: {world_size}"
        )

        assert trial is not None

        if isinstance(trial.study._storage, InMemoryStorage):
            message = "InMemoryStorage is not supported in distributed configuration."
            message += " You have to use RDB or Redis to use Optuna with AllenNLP distributed."
            raise ValueError(message)

        elif isinstance(trial.study._storage, _CachedStorage):
            # Reconstruct storage to purge a old cache.
            if isinstance(trial.study._storage, _CachedStorage):
                trial.study._storage = _CachedStorage(trial.study._storage._backend)

        try:
            mp.spawn(
                _train_worker_with_optuna,
                args=(
                    params.duplicate(),
                    serialization_dir,
                    include_package,
                    dry_run,
                    node_rank,
                    primary_addr,
                    primary_port,
                    world_size,
                    device_ids,
                    file_friendly_logging,
                    include_in_archive,
                    trial,
                ),
                nprocs=num_procs,
            )
        except Exception as e:
            if "optuna.exceptions.TrialPruned" in str(e):
                raise TrialPruned()
            raise e

        if dry_run:
            return None
        else:
            archive_model(serialization_dir, include_in_archive=include_in_archive)
            model = Model.load(params, serialization_dir)
            return model


def _train_worker_with_optuna(
    process_rank: int,
    params: Params,
    serialization_dir: Union[str, PathLike],
    include_package: List[str] = None,
    dry_run: bool = False,
    node_rank: int = 0,
    primary_addr: str = "127.0.0.1",
    primary_port: int = 29500,
    world_size: int = 1,
    distributed_device_ids: List[int] = None,
    file_friendly_logging: bool = False,
    include_in_archive: List[str] = None,
    trial: Trial = None,
) -> Optional[Model]:
    """Extended AllenNLP train_worker.

    Args:
        process_rank : `int`
            The process index that is initialized using the GPU device id.
        params : `Params`
            A parameter object specifying an AllenNLP Experiment.
        serialization_dir : `str`
            The directory in which to save results and logs.
        include_package : `List[str]`, optional
            In distributed mode, since this function would have been spawned as a separate process,
            the extra imports need to be done again. NOTE: This does not have any effect in single
            GPU training.
        dry_run : `bool`, optional (default=`False`)
            Do not train a model, but create a vocabulary, show dataset statistics and other
            training information.
        node_rank : `int`, optional
            Rank of the node.
        primary_addr : `str`, optional (default=`"127.0.0.1"`)
            Address of the primary node for distributed training.
        primary_port : `str`, optional (default=`"29500"`)
            Port of the primary node for distributed training.
        world_size : `int`, optional
            The number of processes involved in distributed training.
        distributed_device_ids: `List[str]`, optional
            IDs of the devices used involved in distributed training.
        file_friendly_logging : `bool`, optional (default=`False`)
            If `True`, we add newlines to tqdm output, even on an interactive terminal, and we slow
            down tqdm's output to only once every 10 seconds.
        include_in_archive : `List[str]`, optional
            Paths relative to `serialization_dir` that should be archived
            in addition to the default ones.
    """

    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    common_logging.prepare_global_logging(
        serialization_dir,
        rank=process_rank,
        world_size=world_size,
    )
    common_util.prepare_environment(params)

    distributed = world_size > 1

    primary = process_rank == 0

    include_package = include_package or []

    if distributed:
        assert distributed_device_ids is not None

        # Since the worker is spawned and not forked, the extra imports need to be done again.
        # Both the ones from the plugins and the ones from `include_package`.
        import_plugins()
        for package_name in include_package:
            common_util.import_module_and_submodules(package_name)

        num_procs_per_node = len(distributed_device_ids)
        # The Unique identifier of the worker process among all the processes in the
        # distributed training group is computed here. This is used while initializing
        # the process group using `init_process_group`
        global_rank = node_rank * num_procs_per_node + process_rank

        # Number of processes per node is useful to know if a process
        # is a primary in the local node(node in which it is running)
        os.environ["ALLENNLP_PROCS_PER_NODE"] = str(num_procs_per_node)

        # In distributed training, the configured device is always going to be a list.
        # The corresponding gpu id for the particular worker is obtained by picking the id
        # from the device list with the rank as index
        gpu_id = distributed_device_ids[process_rank]  # type: ignore

        # Till now, "cuda_device" might not be set in the trainer params.
        # But a worker trainer needs to only know about its specific GPU id.
        params["trainer"]["cuda_device"] = gpu_id
        params["trainer"]["world_size"] = world_size
        params["trainer"]["distributed"] = True

        if gpu_id >= 0:
            torch.cuda.set_device(int(gpu_id))
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{primary_addr}:{primary_port}",
                world_size=world_size,
                rank=global_rank,
            )
        else:
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://{primary_addr}:{primary_port}",
                world_size=world_size,
                rank=global_rank,
            )
        logger.info(
            f"Process group of world size {world_size} initialized "
            f"for distributed training in worker {global_rank}"
        )

    train_loop = TrainModel.from_params(
        params=params,
        serialization_dir=serialization_dir,
        local_rank=process_rank,
        trial=trial,
    )

    if dry_run:
        return None

    try:
        if distributed:  # let the setup get ready for all the workers
            dist.barrier()

        metrics = train_loop.run()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if primary and os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            if hasattr(train_loop.trainer, "get_best_weights_path"):
                best_weights_path = train_loop.trainer.get_best_weights_path  # type: Any
                best_weights_path = cast(Optional[str], best_weights_path)

                if best_weights_path is None:
                    logger.info(
                        "Training interrupted by the user, and no best model has been saved. "
                        "No model archive created."
                    )
                else:
                    logger.info(
                        "Training interrupted by the user. Attempting to create "
                        "a model archive using the current best epoch weights."
                    )
                    archive_model(
                        serialization_dir,
                        weights=best_weights_path,
                        include_in_archive=include_in_archive,
                    )

            else:
                # TODO(himkt): Remove this fallback logic.
                # During AllenNLP 2.x.0, we both support
                # the old implementation and the latest implementation.
                # If we decide to drop support AllenNLP older than 2.5.0, remove here.
                #
                # ref. https://github.com/allenai/allennlp/pull/5220
                #      https://github.com/allenai/allennlp/releases/tag/v2.5.0
                logger.info(
                    "Training interrupted by the user. Attempting to create "
                    "a model archive using the current best epoch weights."
                )
                archive_model(serialization_dir, include_in_archive=include_in_archive)
        raise

    if primary:
        train_loop.finish(metrics)

    if not distributed:
        return train_loop.model

    return None
