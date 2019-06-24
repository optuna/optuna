from __future__ import absolute_import

from argparse import ArgumentParser  # NOQA
from argparse import Namespace  # NOQA
from cliff.app import App
from cliff.command import Command
from cliff.commandmanager import CommandManager
from cliff.lister import Lister
import imp
import logging
import sys

import optuna
from optuna.storages import RDBStorage
from optuna.structs import CLIUsageError
from optuna import types

if types.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA


def get_storage_url(storage_url, config):
    # type: (Optional[str], optuna.config.OptunaConfig) -> str

    if storage_url is not None:
        return storage_url

    if config.default_storage is None:
        raise CLIUsageError(
            'Storage URL is specified neither in config file nor --storage option.')

    return config.default_storage


class BaseCommand(Command):
    def __init__(self, *args, **kwargs):
        # type: (List[Any], Dict[str, Any]) -> None

        super(BaseCommand, self).__init__(*args, **kwargs)
        self.logger = optuna.logging.get_logger(__name__)


class CreateStudy(BaseCommand):
    def get_parser(self, prog_name):
        # type: (str) -> ArgumentParser

        parser = super(CreateStudy, self).get_parser(prog_name)
        parser.add_argument(
            '--study-name',
            default=None,
            help='A human-readable name of a study to distinguish it from others.')
        parser.add_argument(
            '--direction',
            type=str,
            choices=('minimize', 'maximize'),
            default='minimize',
            help='Set direction of optimization to a new study. Set \'minimize\' '
            'for minimization and \'maximize\' for maximization.')
        parser.add_argument(
            '--skip-if-exists',
            default=False,
            action='store_true',
            help='If specified, the creation of the study is skipped '
            'without any error when the study name is duplicated.')
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> None

        config = optuna.config.load_optuna_config(self.app_args.config)
        storage_url = get_storage_url(self.app_args.storage, config)
        storage = optuna.storages.RDBStorage(storage_url)
        study_name = optuna.create_study(
            storage,
            study_name=parsed_args.study_name,
            direction=parsed_args.direction,
            load_if_exists=parsed_args.skip_if_exists).study_name
        print(study_name)


class StudySetUserAttribute(BaseCommand):
    def get_parser(self, prog_name):
        # type: (str) -> ArgumentParser

        parser = super(StudySetUserAttribute, self).get_parser(prog_name)
        parser.add_argument('--study', required=True, help='Study name.')
        parser.add_argument('--key', '-k', required=True, help='Key of the user attribute.')
        parser.add_argument('--value', '-v', required=True, help='Value to be set.')
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> None

        config = optuna.config.load_optuna_config(self.app_args.config)
        storage_url = get_storage_url(self.app_args.storage, config)
        study = optuna.load_study(storage=storage_url, study_name=parsed_args.study)
        study.set_user_attr(parsed_args.key, parsed_args.value)

        self.logger.info('Attribute successfully written.')


class Studies(Lister):

    _datetime_format = '%Y-%m-%d %H:%M:%S'
    _study_list_header = ('NAME', 'DIRECTION', 'N_TRIALS', 'DATETIME_START')

    def get_parser(self, prog_name):
        # type: (str) -> ArgumentParser

        parser = super(Studies, self).get_parser(prog_name)
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> Tuple[Tuple, Tuple[Tuple, ...]]

        config = optuna.config.load_optuna_config(self.app_args.config)
        storage_url = get_storage_url(self.app_args.storage, config)
        summaries = optuna.get_all_study_summaries(storage=storage_url)

        rows = []
        for s in summaries:
            start = s.datetime_start.strftime(self._datetime_format) \
                if s.datetime_start is not None else None
            row = (s.study_name, s.direction.name, s.n_trials, start)
            rows.append(row)

        return self._study_list_header, tuple(rows)


class Dashboard(BaseCommand):
    def get_parser(self, prog_name):
        # type: (str) -> ArgumentParser

        parser = super(Dashboard, self).get_parser(prog_name)
        parser.add_argument('--study', required=True, help='Study name.')
        parser.add_argument(
            '--out',
            '-o',
            help='Output HTML file path. If it is not given, a HTTP server starts '
            'and the dashboard is served.')
        parser.add_argument(
            '--allow-websocket-origin',
            dest='bokeh_allow_websocket_origins',
            action='append',
            default=[],
            help='Allow websocket access from the specified host(s).'
            'Internally, it is used as the value of bokeh\'s '
            '--allow-websocket-origin option. Please refer to '
            'https://bokeh.pydata.org/en/latest/docs/'
            'reference/command/subcommands/serve.html '
            'for more details.')
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> None

        config = optuna.config.load_optuna_config(self.app_args.config)
        storage_url = get_storage_url(self.app_args.storage, config)
        study = optuna.load_study(storage=storage_url, study_name=parsed_args.study)

        if parsed_args.out is None:
            optuna.dashboard.serve(study, parsed_args.bokeh_allow_websocket_origins)
        else:
            optuna.dashboard.write(study, parsed_args.out)
            self.logger.info('Report successfully written to: {}'.format(parsed_args.out))


class StudyOptimize(BaseCommand):
    def get_parser(self, prog_name):
        # type: (str) -> ArgumentParser

        parser = super(StudyOptimize, self).get_parser(prog_name)
        parser.add_argument(
            '--n-trials',
            type=int,
            help='The number of trials. If this argument is not given, as many '
            'trials run as possible.')
        parser.add_argument(
            '--timeout',
            type=float,
            help='Stop study after the given number of second(s). If this argument'
            ' is not given, as many trials run as possible.')
        parser.add_argument(
            '--n-jobs',
            type=int,
            default=1,
            help='The number of parallel jobs. If this argument is set to -1, the '
            'number is set to CPU counts.')
        parser.add_argument('--study', required=True, help='Study name.')
        parser.add_argument(
            'file', help='Python script file where the objective function resides.')
        parser.add_argument('method', help='The method name of the objective function.')
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> int

        config = optuna.config.load_optuna_config(self.app_args.config)
        storage_url = get_storage_url(self.app_args.storage, config)
        study = optuna.load_study(storage=storage_url, study_name=parsed_args.study)

        # We force enabling the debug flag. As we are going to execute user codes, we want to show
        # exception stack traces by default.
        self.app.options.debug = True

        target_module = imp.load_source('optuna_target_module', parsed_args.file)

        try:
            target_method = getattr(target_module, parsed_args.method)
        except AttributeError:
            self.logger.error('Method {} not found in file {}.'.format(
                parsed_args.method, parsed_args.file))
            return 1

        study.optimize(
            target_method,
            n_trials=parsed_args.n_trials,
            timeout=parsed_args.timeout,
            n_jobs=parsed_args.n_jobs)
        return 0


class StorageUpgrade(BaseCommand):
    def get_parser(self, prog_name):
        # type: (str) -> ArgumentParser

        parser = super(StorageUpgrade, self).get_parser(prog_name)
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> None

        if self.app_args.storage is None and self.app_args.config is None:
            raise CLIUsageError("Either --storage or --config option is required.")

        config = optuna.config.load_optuna_config(self.app_args.config)
        storage_url = get_storage_url(self.app_args.storage, config)

        storage = RDBStorage(storage_url, skip_compatibility_check=True)
        current_version = storage.get_current_version()
        head_version = storage.get_head_version()
        known_versions = storage.get_all_versions()
        if current_version == head_version:
            self.logger.info('This storage is up-to-date.')
        elif current_version in known_versions:
            self.logger.info('Upgrading the storage schema to the latest version.')
            storage.upgrade()
            self.logger.info("Completed to upgrade the storage.")
        else:
            self.logger.warning('Your optuna version seems outdated against the storage version. '
                                'Please try updating optuna to the latest version by '
                                '`$ pip install -U optuna`.')


_COMMANDS = {
    'create-study': CreateStudy,
    'study set-user-attr': StudySetUserAttribute,
    'studies': Studies,
    'dashboard': Dashboard,
    'study optimize': StudyOptimize,
    'storage upgrade': StorageUpgrade,
}


class OptunaApp(App):
    def __init__(self):
        # type: () -> None

        command_manager = CommandManager('optuna.command')
        super(OptunaApp, self).__init__(
            description='', version=optuna.__version__, command_manager=command_manager)
        for name, cls in _COMMANDS.items():
            command_manager.add_command(name, cls)

    def build_option_parser(self, description, version, argparse_kwargs=None):
        # type: (str, str, Optional[Dict]) -> ArgumentParser

        parser = super(OptunaApp, self).build_option_parser(description, version, argparse_kwargs)
        parser.add_argument(
            '--config', default=None, help='Config file path. (default=$HOME/.optuna.yml)')
        parser.add_argument('--storage', default=None, help='DB URL. (e.g. sqlite:///example.db)')
        return parser

    def configure_logging(self):
        # type: () -> None

        super(OptunaApp, self).configure_logging()

        # Find the StreamHandler that is configured by super's configure_logging,
        # and replace its formatter with our fancy one.
        root_logger = logging.getLogger()
        stream_handlers = [
            handler for handler in root_logger.handlers
            if isinstance(handler, logging.StreamHandler)
        ]
        assert len(stream_handlers) == 1
        stream_handler = stream_handlers[0]
        stream_handler.setFormatter(optuna.logging.create_default_formatter())

    def clean_up(self, cmd, result, err):
        # type: (Command, int, Optional[Exception]) -> None

        if isinstance(err, CLIUsageError):
            self.parser.print_help()


def main():
    # type: () -> int

    argv = sys.argv[1:] if len(sys.argv) > 1 else ['help']
    return OptunaApp().run(argv)
