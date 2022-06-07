"""Optuna CLI module.

This module is implemented using cliff. It follows
[the demoapp](https://docs.openstack.org/cliff/latest/user/demoapp.html).

If you want to add a new command, you also need to update `entry_points` in `setup.py`.
c.f. https://docs.openstack.org/cliff/latest/user/demoapp.html#setup-py
"""

from argparse import ArgumentParser  # NOQA
from argparse import Namespace  # NOQA
import datetime
from enum import Enum
from importlib.machinery import SourceFileLoader
import json
import logging
import sys
import types
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import warnings

from cliff.app import App
from cliff.command import Command
from cliff.commandmanager import CommandManager
import yaml

import optuna
from optuna._imports import _LazyImport
from optuna.exceptions import CLIUsageError
from optuna.exceptions import ExperimentalWarning
from optuna.storages import RDBStorage
from optuna.trial import TrialState


_dataframe = _LazyImport("optuna.study._dataframe")

_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _check_storage_url(storage_url: Optional[str]) -> str:

    if storage_url is None:
        raise CLIUsageError("Storage URL is not specified.")
    return storage_url


def _format_value(value: Any) -> Any:
    #  Format value that can be serialized to JSON or YAML.
    if value is None or isinstance(value, (int, float)):
        return value
    elif isinstance(value, datetime.datetime):
        return value.strftime(_DATETIME_FORMAT)
    elif isinstance(value, list):
        return list(_format_value(v) for v in value)
    elif isinstance(value, tuple):
        return tuple(_format_value(v) for v in value)
    elif isinstance(value, dict):
        return {_format_value(k): _format_value(v) for k, v in value.items()}
    else:
        return str(value)


def _convert_to_dict(
    records: List[Dict[Tuple[str, str], Any]], columns: List[Tuple[str, str]], flatten: bool
) -> Tuple[List[Dict[str, Any]], List[str]]:
    header = []
    ret = []
    if flatten:
        for column in columns:
            if column[1] != "":
                header.append(f"{column[0]}_{column[1]}")
            elif any(isinstance(record.get(column), (list, tuple)) for record in records):
                max_length = 0
                for record in records:
                    if column in record:
                        max_length = max(max_length, len(record[column]))
                for i in range(max_length):
                    header.append(f"{column[0]}_{i}")
            else:
                header.append(column[0])
        for record in records:
            row = {}
            for column in columns:
                if column not in record:
                    continue
                value = _format_value(record[column])
                if column[1] != "":
                    row[f"{column[0]}_{column[1]}"] = value
                elif any(isinstance(record.get(column), (list, tuple)) for record in records):
                    for i, v in enumerate(value):
                        row[f"{column[0]}_{i}"] = v
                else:
                    row[f"{column[0]}"] = value
            ret.append(row)
    else:
        for column in columns:
            if column[0] not in header:
                header.append(column[0])
        for record in records:
            attrs: Dict[str, Any] = {column_name: {} for column_name in header}
            for column in columns:
                if column not in record:
                    continue
                value = _format_value(record[column])
                if isinstance(column[1], int):
                    # Reconstruct list of values. `_dataframe._create_records_and_aggregate_column`
                    # returns indices of list as the second key of column.
                    if attrs[column[0]] == {}:
                        attrs[column[0]] = []
                    attrs[column[0]] += [None] * max(column[1] + 1 - len(attrs[column[0]]), 0)
                    attrs[column[0]][column[1]] = value
                elif column[1] != "":
                    attrs[column[0]][column[1]] = value
                else:
                    attrs[column[0]] = value
            ret.append(attrs)

    return ret, header


class ValueType(Enum):
    NONE = 0
    NUMERIC = 1
    STRING = 2


class CellValue:
    def __init__(self, value: Any) -> None:
        self.value = value
        if value is None:
            self.value_type = ValueType.NONE
        elif isinstance(value, (int, float)):
            self.value_type = ValueType.NUMERIC
        else:
            self.value_type = ValueType.STRING

    def __str__(self) -> str:
        if isinstance(self.value, datetime.datetime):
            return self.value.strftime(_DATETIME_FORMAT)
        else:
            return str(self.value)

    def width(self) -> int:
        return len(str(self.value))

    def get_string(self, value_type: ValueType, width: int) -> str:
        value = str(self.value)
        if self.value is None:
            return " " * width
        elif value_type == ValueType.NUMERIC:
            return f"{value:>{width}}"
        else:
            return f"{value:<{width}}"


def _dump_table(records: List[Dict[str, Any]], header: List[str]) -> str:
    rows = []
    for record in records:
        row = []
        for column_name in header:
            row.append(CellValue(record.get(column_name)))
        rows.append(row)

    separator = "+"
    header_string = "|"
    rows_string = ["|" for _ in rows]
    for column in range(len(header)):
        value_types = [row[column].value_type for row in rows]
        value_type = ValueType.NUMERIC
        for t in value_types:
            if t == ValueType.STRING:
                value_type = ValueType.STRING
        max_width = max(len(header[column]), max(row[column].width() for row in rows))
        separator += "-" * (max_width + 2) + "+"
        if value_type == ValueType.NUMERIC:
            header_string += f" {header[column]:>{max_width}} |"
        else:
            header_string += f" {header[column]:<{max_width}} |"
        for i, row in enumerate(rows):
            rows_string[i] += " " + row[column].get_string(value_type, max_width) + " |"

    ret = ""
    ret += separator + "\n"
    ret += header_string + "\n"
    ret += separator + "\n"
    ret += "\n".join(rows_string) + "\n"
    ret += separator + "\n"

    return ret


def _format_output(
    records: Union[List[Dict[Tuple[str, str], Any]], Dict[Tuple[str, str], Any]],
    columns: List[Tuple[str, str]],
    output_format: str,
    flatten: bool,
) -> str:
    if isinstance(records, list):
        values, header = _convert_to_dict(records, columns, flatten)
    else:
        values, header = _convert_to_dict([records], columns, flatten)

    if output_format == "table":
        return _dump_table(values, header).strip()
    elif output_format == "json":
        if isinstance(records, list):
            return json.dumps(values).strip()
        else:
            return json.dumps(values[0]).strip()
    elif output_format == "yaml":
        if isinstance(records, list):
            return yaml.safe_dump(values).strip()
        else:
            return yaml.safe_dump(values[0]).strip()
    else:
        raise CLIUsageError(f"Optuna CLI does not supported the {output_format} format.")


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
        parser.add_argument("--value", required=True, help="Value to be set.")
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


class _Studies(_BaseCommand):
    """Show a list of studies."""

    _study_list_header = [
        ("name", ""),
        ("direction", ""),
        ("n_trials", ""),
        ("datetime_start", ""),
    ]

    def get_parser(self, prog_name: str) -> ArgumentParser:

        parser = super(_Studies, self).get_parser(prog_name)
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            choices=("json", "table", "yaml"),
            default="table",
            help="Output format.",
        )
        parser.add_argument(
            "--flatten",
            default=False,
            action="store_true",
            help="Flatten nested columns such as directions.",
        )
        return parser

    def take_action(self, parsed_args: Namespace) -> None:

        storage_url = _check_storage_url(self.app_args.storage)
        summaries = optuna.get_all_study_summaries(storage_url, include_best_trial=False)

        records = []
        for s in summaries:
            start = (
                s.datetime_start.strftime(_DATETIME_FORMAT)
                if s.datetime_start is not None
                else None
            )
            record: Dict[Tuple[str, str], Any] = {}
            record[("name", "")] = s.study_name
            record[("direction", "")] = tuple(d.name for d in s.directions)
            record[("n_trials", "")] = s.n_trials
            record[("datetime_start", "")] = start
            records.append(record)

        print(
            _format_output(
                records, self._study_list_header, parsed_args.format, parsed_args.flatten
            )
        )


class _Trials(_BaseCommand):
    """Show a list of trials."""

    def get_parser(self, prog_name: str) -> ArgumentParser:

        parser = super(_Trials, self).get_parser(prog_name)
        parser.add_argument(
            "--study-name",
            type=str,
            required=True,
            help="The name of the study which includes trials.",
        )
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            choices=("json", "table", "yaml"),
            default="table",
            help="Output format.",
        )
        parser.add_argument(
            "--flatten",
            default=False,
            action="store_true",
            help="Flatten nested columns such as params and user_attrs.",
        )
        return parser

    def take_action(self, parsed_args: Namespace) -> None:

        warnings.warn(
            "'trials' is an experimental CLI command. The interface can change in the future.",
            ExperimentalWarning,
        )

        storage_url = _check_storage_url(self.app_args.storage)
        study = optuna.load_study(storage=storage_url, study_name=parsed_args.study_name)
        attrs = (
            "number",
            "value" if not study._is_multi_objective() else "values",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "state",
        )

        records, columns = _dataframe._create_records_and_aggregate_column(study, attrs)
        print(_format_output(records, columns, parsed_args.format, parsed_args.flatten))


class _BestTrial(_BaseCommand):
    """Show the best trial."""

    def get_parser(self, prog_name: str) -> ArgumentParser:

        parser = super(_BestTrial, self).get_parser(prog_name)
        parser.add_argument(
            "--study-name",
            type=str,
            required=True,
            help="The name of the study to get the best trial.",
        )
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            choices=("json", "table", "yaml"),
            default="table",
            help="Output format.",
        )
        parser.add_argument(
            "--flatten",
            default=False,
            action="store_true",
            help="Flatten nested columns such as params and user_attrs.",
        )
        return parser

    def take_action(self, parsed_args: Namespace) -> None:

        warnings.warn(
            "'best-trial' is an experimental CLI command. The interface can change in the future.",
            ExperimentalWarning,
        )

        storage_url = _check_storage_url(self.app_args.storage)
        study = optuna.load_study(storage=storage_url, study_name=parsed_args.study_name)
        attrs = (
            "number",
            "value" if not study._is_multi_objective() else "values",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "state",
        )

        records, columns = _dataframe._create_records_and_aggregate_column(study, attrs)
        print(
            _format_output(
                records[study.best_trial.number], columns, parsed_args.format, parsed_args.flatten
            )
        )


class _BestTrials(_BaseCommand):
    """Show a list of trials located at the Pareto front."""

    def get_parser(self, prog_name: str) -> ArgumentParser:

        parser = super(_BestTrials, self).get_parser(prog_name)
        parser.add_argument(
            "--study-name",
            type=str,
            required=True,
            help="The name of the study to get the best trials (trials at the Pareto front).",
        )
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            choices=("json", "table", "yaml"),
            default="table",
            help="Output format.",
        )
        parser.add_argument(
            "--flatten",
            default=False,
            action="store_true",
            help="Flatten nested columns such as params and user_attrs.",
        )
        return parser

    def take_action(self, parsed_args: Namespace) -> None:

        warnings.warn(
            "'best-trials' is an experimental CLI command. The interface can change in the "
            "future.",
            ExperimentalWarning,
        )

        storage_url = _check_storage_url(self.app_args.storage)
        study = optuna.load_study(storage=storage_url, study_name=parsed_args.study_name)
        best_trials = [trial.number for trial in study.best_trials]
        attrs = (
            "number",
            "value" if not study._is_multi_objective() else "values",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "state",
        )

        records, columns = _dataframe._create_records_and_aggregate_column(study, attrs)
        best_records = list(filter(lambda record: record[("number", "")] in best_trials, records))
        print(_format_output(best_records, columns, parsed_args.format, parsed_args.flatten))


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
        storage = RDBStorage(storage_url, skip_compatibility_check=True, skip_table_creation=True)
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
            help=(
                "Direction of optimization. This argument is deprecated."
                " Please create a study in advance."
            ),
        )
        parser.add_argument(
            "--directions",
            type=str,
            nargs="+",
            choices=("minimize", "maximize"),
            help=(
                "Directions of optimization, if there are multiple objectives."
                " This argument is deprecated. Please create a study in advance."
            ),
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
            "-f",
            "--format",
            type=str,
            choices=("json", "table", "yaml"),
            default="json",
            help="Output format.",
        )
        parser.add_argument(
            "--flatten",
            default=False,
            action="store_true",
            help="Flatten nested columns such as params.",
        )
        return parser

    def take_action(self, parsed_args: Namespace) -> None:

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

        if parsed_args.direction is not None or parsed_args.directions is not None:
            message = (
                "The `direction` and `directions` arguments of the `study ask` command are"
                " deprecated because the command will no longer create a study when you specify"
                " the arguments. Please create a study in advance."
            )
            warnings.warn(message, FutureWarning)

        if parsed_args.sampler is not None:
            if parsed_args.sampler_kwargs is not None:
                sampler_kwargs = json.loads(parsed_args.sampler_kwargs)
            else:
                sampler_kwargs = {}
            sampler_cls = getattr(optuna.samplers, parsed_args.sampler)
            sampler = sampler_cls(**sampler_kwargs)
            create_study_kwargs["sampler"] = sampler
        else:
            if parsed_args.sampler_kwargs is not None:
                raise ValueError(
                    "`--sampler_kwargs` is set without `--sampler`. Please specify `--sampler` as"
                    " well or omit `--sampler-kwargs`."
                )

        if parsed_args.search_space is not None:
            # The search space is expected to be a JSON serialized string, e.g.
            # '{"x": {"name": "FloatDistribution", "attributes": {"low": 0.0, "high": 1.0}},
            #   "y": ...}'.
            search_space = {
                name: optuna.distributions.json_to_distribution(json.dumps(dist))
                for name, dist in json.loads(parsed_args.search_space).items()
            }
        else:
            search_space = {}

        try:
            study = optuna.load_study(
                study_name=create_study_kwargs["study_name"],
                storage=create_study_kwargs["storage"],
                sampler=create_study_kwargs.get("sampler"),
            )
            directions = None
            if (
                create_study_kwargs["direction"] is not None
                and create_study_kwargs["directions"] is not None
            ):
                raise ValueError("Specify only one of `direction` and `directions`.")
            if create_study_kwargs["direction"] is not None:
                directions = [
                    optuna.study.StudyDirection[create_study_kwargs["direction"].upper()]
                ]
            if create_study_kwargs["directions"] is not None:
                directions = [
                    optuna.study.StudyDirection[d.upper()]
                    for d in create_study_kwargs["directions"]
                ]
            if directions is not None and study.directions != directions:
                raise ValueError(
                    f"Cannot overwrite study direction from {study.directions} to {directions}."
                )

        except KeyError:
            study = optuna.create_study(**create_study_kwargs)
        trial = study.ask(fixed_distributions=search_space)

        self.logger.info(f"Asked trial {trial.number} with parameters {trial.params}.")

        record: Dict[Tuple[str, str], Any] = {("number", ""): trial.number}
        columns = [("number", "")]

        if len(trial.params) == 0 and not parsed_args.flatten:
            record[("params", "")] = {}
            columns.append(("params", ""))
        else:
            for param_name, param_value in trial.params.items():
                record[("params", param_name)] = param_value
                columns.append(("params", param_name))

        print(_format_output(record, columns, parsed_args.format, parsed_args.flatten))


class _Tell(_BaseCommand):
    """Finish a trial, which was created by the ask command."""

    def get_parser(self, prog_name: str) -> ArgumentParser:

        parser = super(_Tell, self).get_parser(prog_name)
        parser.add_argument("--study-name", type=str, help="Name of study.")
        parser.add_argument("--trial-number", type=int, help="Trial number.")
        parser.add_argument("--values", type=float, nargs="+", help="Objective values.")
        parser.add_argument(
            "--state",
            type=str,
            help="Trial state.",
            choices=("complete", "pruned", "fail"),
        )
        parser.add_argument(
            "--skip-if-finished",
            default=False,
            action="store_true",
            help="If specified, tell is skipped without any error when the trial is already"
            "finished.",
        )
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

        if parsed_args.state is not None:
            state: Optional[TrialState] = TrialState[parsed_args.state.upper()]
        else:
            state = None

        trial_number = parsed_args.trial_number
        values = parsed_args.values

        study.tell(
            trial=trial_number,
            values=values,
            state=state,
            skip_if_finished=parsed_args.skip_if_finished,
        )

        self.logger.info(f"Told trial {trial_number} with values {values} and state {state}.")

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
        optuna.logging.set_verbosity(stream_handler.level)

    def clean_up(self, cmd: Command, result: int, err: Optional[Exception]) -> None:

        if isinstance(err, CLIUsageError):
            self.parser.print_help()


def main() -> int:

    argv = sys.argv[1:] if len(sys.argv) > 1 else ["help"]
    return _OptunaApp().run(argv)
