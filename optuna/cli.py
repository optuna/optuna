"""Optuna CLI module.

This module is implemented using cliff. It follows
[the demoapp](https://docs.openstack.org/cliff/latest/user/demoapp.html).

If you want to add a new command, you also need to update `entry_points` in `setup.py`.
c.f. https://docs.openstack.org/cliff/latest/user/demoapp.html#setup-py
"""

from argparse import ArgumentParser  # NOQA
from argparse import Namespace  # NOQA
from importlib.machinery import SourceFileLoader
import json
import logging
import sys
import types
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
import warnings

from cliff.app import App
from cliff.command import Command
from cliff.commandmanager import CommandManager
from cliff.lister import Lister
import yaml

import optuna
from optuna._deprecated import deprecated
from optuna.exceptions import CLIUsageError
from optuna.exceptions import ExperimentalWarning
from optuna.storages import RDBStorage
from optuna.trial import TrialState


def _check_storage_url(storage_url: Optional[str]) -> str:

    if storage_url is None:
        raise CLIUsageError("Storage URL is not specified.")
    return storage_url


class _BaseCommand(Command):
    def __init__(self, *args: Any, **kwargs: Any) -> None:

        super().__init__(*args, **kwargs)
        self.logger = optuna.logging.get_logger(__name__)


class _CreateStudy(_BaseCommand):
    """Create a new study."""

    def get_parser(self, prog_name: str) -> ArgumentParser:

        parser = super(_CreateStudy, self).get_parser(prog_name)
        parser.add_argument(
            "--study-name",
            default=None,
            help="A human-readable name of a study to distinguish it from others.",
        )
        parser.add_argument(
            "--direction",
            default=None,
            type=str,
            choices=("minimize", "maximize"),
            help="Set direction of optimization to a new study. Set 'minimize' "
            "for minimization and 'maximize' for maximization.",
        )
        parser.add_argument(
            "--skip-if-exists",
            default=False,
            action="store_true",
            help="If specified, the creation of the study is skipped "
            "without any error when the study name is duplicated.",
        )
        parser.add_argument(
            "--directions",
            type=str,
            default=None,
            choices=("minimize", "maximize"),
            help="Set directions of optimization to a new study."
            " Put whitespace between directions. Each direction should be"
            ' either "minimize" or "maximize".',
            nargs="+",
        )
        return parser

    def take_action(self, parsed_args: Namespace) -> None:

        storage_url = _check_storage_url(self.app_args.storage)
        storage = optuna.storages.get_storage(storage_url)
        study_name = optuna.create_study(
            storage=storage,
            study_name=parsed_args.study_name,
            direction=parsed_args.direction,
            directions=parsed_args.directions,
            load_if_exists=parsed_args.skip_if_exists,
        ).study_name
        print(study_name)


class _DeleteStudy(_BaseCommand):
    """Delete a specified study."""

    def get_parser(self, prog_name: str) -> ArgumentParser:

        parser = super(_DeleteStudy, self).get_parser(prog_name)
        parser.add_argument("--study-name", default=None, help="The name of the study to delete.")
        return parser

    def take_action(self, parsed_args: Namespace) -> None:

        storage_url = _check_storage_url(self.app_args.storage)
        storage = optuna.storages.get_storage(storage_url)
        study_id = storage.get_study_id_from_name(parsed_args.study_name)
        storage.delete_study(study_id)


class _StudySetUserAttribute(_BaseCommand):
    """Set a user attribute to a study."""

    def get_parser(self, prog_name: str) -> ArgumentParser:

        parser = super(_StudySetUserAttribute, self).get_parser(prog_name)
        parser.add_argument(
            "--study", default=None, help="This argument is deprecated. Use --study-name instead."
        )
        parser.add_argument(
            "--study-name",
            default=None,
            help="The name of the study to set the user attribute to.",
        )
        parser.add_argument("--key", "-k", required=True, help="Key of the user attribute.")
        parser.add_argument("--value", "-v", required=True, help="Value to be set.")
        return parser

    def take_action(self, parsed_args: Namespace) -> None:

        storage_url = _check_storage_url(self.app_args.storage)

        if parsed_args.study and parsed_args.study_name:
            raise ValueError(
                "Both `--study-name` and the deprecated `--study` was specified. "
                "Please remove the `--study` flag."
            )
        elif parsed_args.study:
            message = "The use of `--study` is deprecated. Please use `--study-name` instead."
            warnings.warn(message, FutureWarning)
            study = optuna.load_study(storage=storage_url, study_name=parsed_args.study)
        elif parsed_args.study_name:
            study = optuna.load_study(storage=storage_url, study_name=parsed_args.study_name)
        else:
            raise ValueError("Missing study name. Please use `--study-name`.")

        study.set_user_attr(parsed_args.key, parsed_args.value)

        self.logger.info("Attribute successfully written.")


class _Studies(Lister):
    """Show a list of studies."""

    _datetime_format = "%Y-%m-%d %H:%M:%S"
    _study_list_header = ("NAME", "DIRECTION", "N_TRIALS", "DATETIME_START")

    def get_parser(self, prog_name: str) -> ArgumentParser:

        parser = super(_Studies, self).get_parser(prog_name)
        return parser

    def take_action(self, parsed_args: Namespace) -> Tuple[Tuple, Tuple[Tuple, ...]]:

        storage_url = _check_storage_url(self.app_args.storage)
        summaries = optuna.get_all_study_summaries(storage=storage_url)

        rows = []
        for s in summaries:
            start = (
                s.datetime_start.strftime(self._datetime_format)
                if s.datetime_start is not None
                else None
            )
            row = (s.study_name, tuple(d.name for d in s.directions), s.n_trials, start)
            rows.append(row)

        return self._study_list_header, tuple(rows)


class _Dashboard(_BaseCommand):
    """Launch web dashboard (beta).

    This feature is deprecated since version 2.7.0. Please use `optuna-dashboard
    <https://github.com/optuna/optuna-dashboard>`_ instead.
    """

    def get_parser(self, prog_name: str) -> ArgumentParser:

        parser = super(_Dashboard, self).get_parser(prog_name)
        parser.add_argument(
            "--study", default=None, help="This argument is deprecated. Use --study-name instead."
        )
        parser.add_argument(
            "--study-name", default=None, help="The name of the study to show on the dashboard."
        )
        parser.add_argument(
            "--out",
            "-o",
            help="Output HTML file path. If it is not given, a HTTP server starts "
            "and the dashboard is served.",
        )
        parser.add_argument(
            "--allow-websocket-origin",
            dest="bokeh_allow_websocket_origins",
            action="append",
            default=[],
            help="Allow websocket access from the specified host(s)."
            "Internally, it is used as the value of bokeh's "
            "--allow-websocket-origin option. Please refer to "
            "https://bokeh.pydata.org/en/latest/docs/"
            "reference/command/subcommands/serve.html "
            "for more details.",
        )
        return parser

    @deprecated(
        "2.7.0",
        "3.0.0",
        name="dashboard",
        text="Please use optuna-dashboard (https://github.com/optuna/optuna-dashboard) instead.",
    )
    def take_action(self, parsed_args: Namespace) -> None:

        storage_url = _check_storage_url(self.app_args.storage)

        if parsed_args.study and parsed_args.study_name:
            raise ValueError(
                "Both `--study-name` and the deprecated `--study` was specified. "
                "Please remove the `--study` flag."
            )
        elif parsed_args.study:
            message = "The use of `--study` is deprecated. Please use `--study-name` instead."
            warnings.warn(message, FutureWarning)
            study = optuna.load_study(storage=storage_url, study_name=parsed_args.study)
        elif parsed_args.study_name:
            study = optuna.load_study(storage=storage_url, study_name=parsed_args.study_name)
        else:
            raise ValueError("Missing study name. Please use `--study-name`.")

        if parsed_args.out is None:
            optuna.dashboard._serve(study, parsed_args.bokeh_allow_websocket_origins)
        else:
            optuna.dashboard._write(study, parsed_args.out)
            self.logger.info("Report successfully written to: {}".format(parsed_args.out))


class _StudyOptimize(_BaseCommand):
    """Start optimization of a study. Deprecated since version 2.0.0."""

    def get_parser(self, prog_name: str) -> ArgumentParser:

        parser = super(_StudyOptimize, self).get_parser(prog_name)
        parser.add_argument(
            "--n-trials",
            type=int,
            help="The number of trials. If this argument is not given, as many "
            "trials run as possible.",
        )
        parser.add_argument(
            "--timeout",
            type=float,
            help="Stop study after the given number of second(s). If this argument"
            " is not given, as many trials run as possible.",
        )
        parser.add_argument(
            "--n-jobs",
            type=int,
            default=1,
            help="The number of parallel jobs. If this argument is set to -1, the "
            "number is set to CPU counts.",
        )
        parser.add_argument(
            "--study", default=None, help="This argument is deprecated. Use --study-name instead."
        )
        parser.add_argument(
            "--study-name", default=None, help="The name of the study to start optimization on."
        )
        parser.add_argument(
            "file", help="Python script file where the objective function resides."
        )
        parser.add_argument("method", help="The method name of the objective function.")
        return parser

    def take_action(self, parsed_args: Namespace) -> int:

        message = (
            "The use of the `study optimize` command is deprecated. Please execute your Python "
            "script directly instead."
        )
        warnings.warn(message, FutureWarning)

        storage_url = _check_storage_url(self.app_args.storage)

        if parsed_args.study and parsed_args.study_name:
            raise ValueError(
                "Both `--study-name` and the deprecated `--study` was specified. "
                "Please remove the `--study` flag."
            )
        elif parsed_args.study:
            message = "The use of `--study` is deprecated. Please use `--study-name` instead."
            warnings.warn(message, FutureWarning)
            study = optuna.load_study(storage=storage_url, study_name=parsed_args.study)
        elif parsed_args.study_name:
            study = optuna.load_study(storage=storage_url, study_name=parsed_args.study_name)
        else:
            raise ValueError("Missing study name. Please use `--study-name`.")

        # We force enabling the debug flag. As we are going to execute user codes, we want to show
        # exception stack traces by default.
        self.app.options.debug = True

        module_name = "optuna_target_module"
        target_module = types.ModuleType(module_name)
        loader = SourceFileLoader(module_name, parsed_args.file)
        loader.exec_module(target_module)

        try:
            target_method = getattr(target_module, parsed_args.method)
        except AttributeError:
            self.logger.error(
                "Method {} not found in file {}.".format(parsed_args.method, parsed_args.file)
            )
            return 1

        study.optimize(
            target_method,
            n_trials=parsed_args.n_trials,
            timeout=parsed_args.timeout,
            n_jobs=parsed_args.n_jobs,
        )
        return 0


class _StorageUpgrade(_BaseCommand):
    """Upgrade the schema of a storage."""

    def get_parser(self, prog_name: str) -> ArgumentParser:

        parser = super(_StorageUpgrade, self).get_parser(prog_name)
        return parser

    def take_action(self, parsed_args: Namespace) -> None:

        storage_url = _check_storage_url(self.app_args.storage)
        if storage_url.startswith("redis"):
            self.logger.info("This storage does not support upgrade yet.")
            return
        storage = RDBStorage(storage_url, skip_compatibility_check=True)
        current_version = storage.get_current_version()
        head_version = storage.get_head_version()
        known_versions = storage.get_all_versions()
        if current_version == head_version:
            self.logger.info("This storage is up-to-date.")
        elif current_version in known_versions:
            self.logger.info("Upgrading the storage schema to the latest version.")
            storage.upgrade()
            self.logger.info("Completed to upgrade the storage.")
        else:
            warnings.warn(
                "Your optuna version seems outdated against the storage version. "
                "Please try updating optuna to the latest version by "
                "`$ pip install -U optuna`."
            )


class _Ask(_BaseCommand):
    """Create a new trial and suggest parameters."""

    def get_parser(self, prog_name: str) -> ArgumentParser:

        parser = super(_Ask, self).get_parser(prog_name)
        parser.add_argument("--study-name", type=str, help="Name of study.")
        parser.add_argument(
            "--direction",
            type=str,
            choices=("minimize", "maximize"),
            help="Direction of optimization.",
        )
        parser.add_argument(
            "--directions",
            type=str,
            nargs="+",
            choices=("minimize", "maximize"),
            help="Directions of optimization, if there are multiple objectives.",
        )
        parser.add_argument("--sampler", type=str, help="Class name of sampler object to create.")
        parser.add_argument(
            "--sampler-kwargs",
            type=str,
            help="Sampler object initialization keyword arguments as JSON.",
        )
        parser.add_argument(
            "--search-space",
            type=str,
            help=(
                "Search space as JSON. Keys are names and values are outputs from "
                ":func:`~optuna.distributions.distribution_to_json`."
            ),
        )
        parser.add_argument(
            "--out", type=str, choices=("json", "yaml"), default="json", help="Output format."
        )
        return parser

    def take_action(self, parsed_args: Namespace) -> int:

        warnings.warn(
            "'ask' is an experimental CLI command. The interface can change in the future.",
            ExperimentalWarning,
        )

        storage_url = _check_storage_url(self.app_args.storage)

        create_study_kwargs = {
            "storage": storage_url,
            "study_name": parsed_args.study_name,
            "direction": parsed_args.direction,
            "directions": parsed_args.directions,
            "load_if_exists": True,
        }
        if parsed_args.sampler is not None:
            if parsed_args.sampler_kwargs is not None:
                sampler_kwargs = json.loads(parsed_args.sampler_kwargs)
            else:
                sampler_kwargs = {}
            sampler_cls = getattr(optuna.samplers, parsed_args.sampler)
            sampler = sampler_cls(**sampler_kwargs)
            create_study_kwargs["sampler"] = sampler

        if parsed_args.search_space is not None:
            # The search space is expected to be a JSON serialized string, e.g.
            # '{"x": {"name": "UniformDistribution", "attributes": {"low": 0.0, "high": 1.0}},
            #   "y": ...}'.
            search_space = {
                name: optuna.distributions.json_to_distribution(json.dumps(dist))
                for name, dist in json.loads(parsed_args.search_space).items()
            }
        else:
            search_space = {}

        study = optuna.create_study(**create_study_kwargs)
        trial = study.ask(fixed_distributions=search_space)
        out = {
            "trial": {
                "number": trial.number,
                "params": trial.params,
            }
        }

        self.logger.info(f"Asked trial {trial.number} with parameters {trial.params}.")

        if parsed_args.out == "json":
            print(json.dumps(out))
        elif parsed_args.out == "yaml":
            print(yaml.dump(out))
        else:
            assert False

        return 0


class _Tell(_BaseCommand):
    """Finish a trial created with the ask command."""

    def get_parser(self, prog_name: str) -> ArgumentParser:

        parser = super(_Tell, self).get_parser(prog_name)
        parser.add_argument("--study-name", type=str, help="Name of study.")
        parser.add_argument("--trial-number", type=int, help="Trial number.")
        parser.add_argument("--values", type=float, nargs="+", help="Objective values.")
        parser.add_argument("--state", type=str, default="complete", help="Trial state.")
        return parser

    def take_action(self, parsed_args: Namespace) -> int:

        warnings.warn(
            "'tell' is an experimental CLI command. The interface can change in the future.",
            ExperimentalWarning,
        )

        storage_url = _check_storage_url(self.app_args.storage)

        study = optuna.load_study(
            storage=storage_url,
            study_name=parsed_args.study_name,
        )

        state = TrialState[parsed_args.state.upper()]
        study.tell(
            trial=parsed_args.trial_number,
            values=parsed_args.values,
            state=state,
        )

        self.logger.info(
            f"Told trial {parsed_args.trial_number} with values {parsed_args.values} and state "
            f"{state}."
        )

        return 0


class _OptunaApp(App):
    def __init__(self) -> None:

        super().__init__(
            description="",
            version=optuna.__version__,
            command_manager=CommandManager("optuna.command"),
            deferred_help=True,
        )

    def build_option_parser(
        self, description: str, version: str, argparse_kwargs: Optional[Dict] = None
    ) -> ArgumentParser:

        parser = super(_OptunaApp, self).build_option_parser(description, version, argparse_kwargs)
        parser.add_argument("--storage", default=None, help="DB URL. (e.g. sqlite:///example.db)")
        return parser

    def configure_logging(self) -> None:

        super(_OptunaApp, self).configure_logging()

        # Find the StreamHandler that is configured by super's configure_logging,
        # and replace its formatter with our fancy one.
        root_logger = logging.getLogger()
        stream_handlers = [
            handler
            for handler in root_logger.handlers
            if isinstance(handler, logging.StreamHandler)
        ]
        assert len(stream_handlers) == 1
        stream_handler = stream_handlers[0]
        stream_handler.setFormatter(optuna.logging.create_default_formatter())

    def clean_up(self, cmd: Command, result: int, err: Optional[Exception]) -> None:

        if isinstance(err, CLIUsageError):
            self.parser.print_help()


def main() -> int:

    argv = sys.argv[1:] if len(sys.argv) > 1 else ["help"]
    return _OptunaApp().run(argv)
