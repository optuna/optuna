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
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Tuple  # NOQA

import pfnopt


_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
_STUDY_LIST_HEADER = ('UUID', 'TASK', 'N_TRIALS', 'DATETIME_START')


class BaseCommand(Command):

    def __init__(self, *args, **kwargs):
        # type: (List[Any], Dict[str, Any]) -> None

        super(BaseCommand, self).__init__(*args, **kwargs)
        self.logger = pfnopt.logging.get_logger(__name__)


class CreateStudy(BaseCommand):

    def get_parser(self, prog_name):
        # type: (str) -> ArgumentParser

        parser = super(CreateStudy, self).get_parser(prog_name)
        parser.add_argument('--storage', required=True)
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> None

        storage = pfnopt.storages.RDBStorage(parsed_args.storage)
        study_uuid = pfnopt.create_study(storage).study_uuid
        print(study_uuid)


class StudySetUserAttribute(BaseCommand):

    def get_parser(self, prog_name):
        # type: (str) -> ArgumentParser

        parser = super(StudySetUserAttribute, self).get_parser(prog_name)
        parser.add_argument('--storage', required=True)
        parser.add_argument('--study', required=True)
        parser.add_argument('--key', '-k', required=True)
        parser.add_argument('--value', '-v', required=True)
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> None

        study = pfnopt.Study(storage=parsed_args.storage, study_uuid=parsed_args.study)
        study.set_user_attr(parsed_args.key, parsed_args.value)

        self.logger.info('Attribute successfully written.')


class StudyList(Lister):

    def get_parser(self, prog_name):
        # type: (str) -> ArgumentParser

        parser = super(StudyList, self).get_parser(prog_name)
        parser.add_argument('--storage', required=True)
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> Tuple[Tuple, Tuple[Tuple, ...]]

        summaries = pfnopt.get_all_study_summaries(storage=parsed_args.storage)

        rows = []
        for s in summaries:
            start = s.datetime_start if s.datetime_start is not None else None
            row = (s.study_uuid, s.task.name, s.n_trials, start)
            rows.append(row)

        return _STUDY_LIST_HEADER, tuple(rows)


class Dashboard(BaseCommand):

    def get_parser(self, prog_name):
        # type: (str) -> ArgumentParser

        parser = super(Dashboard, self).get_parser(prog_name)
        parser.add_argument('--storage', required=True)
        parser.add_argument('--study', required=True)
        parser.add_argument('--out', '-o')
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> None

        study = pfnopt.Study(storage=parsed_args.storage, study_uuid=parsed_args.study)

        if parsed_args.out is None:
            pfnopt.dashboard.serve(study)
        else:
            pfnopt.dashboard.write(study, parsed_args.out)
            self.logger.info('Report successfully written to: {}'.format(parsed_args.out))


class Minimize(BaseCommand):

    def get_parser(self, prog_name):
        # type: (str) -> ArgumentParser

        parser = super(Minimize, self).get_parser(prog_name)
        parser.add_argument('--n-trials', type=int)
        parser.add_argument('--timeout', type=float)
        parser.add_argument('--n-jobs', type=int, default=1)
        parser.add_argument('--storage')
        parser.add_argument('--study')
        parser.add_argument('--create-study', action='store_true')
        parser.add_argument('file')
        parser.add_argument('method')
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> int

        if parsed_args.create_study and parsed_args.study:
            raise ValueError('Inconsistent arguments. Flags --create-study and --study '
                             'should not be specified at the same time.')
        if not parsed_args.create_study and not parsed_args.study:
            raise ValueError('Inconsistent arguments. Either --create-study or --study '
                             'should be specified.')

        if parsed_args.create_study:
            study = pfnopt.create_study(storage=parsed_args.storage)
        else:
            study = pfnopt.Study(storage=parsed_args.storage, study_uuid=parsed_args.study)

        # We force enabling the debug flag. As we are going to execute user codes, we want to show
        # exception stack traces by default.
        self.app.options.debug = True

        target_module = imp.load_source('pfnopt_target_module', parsed_args.file)

        try:
            target_method = getattr(target_module, parsed_args.method)
        except AttributeError:
            self.logger.error('Method {} not found in file {}.'.format(
                parsed_args.method, parsed_args.file))
            return 1

        pfnopt.minimize(
            target_method, n_trials=parsed_args.n_trials,
            timeout=parsed_args.timeout, n_jobs=parsed_args.n_jobs,
            study=study)
        return 0


_COMMANDS = {
    'create-study': CreateStudy,
    'study set-user-attr': StudySetUserAttribute,
    'study list': StudyList,
    'dashboard': Dashboard,
    'minimize': Minimize,
}


class PFNOptApp(App):

    def __init__(self):
        # type: () -> None

        command_manager = CommandManager('pfnopt.command')
        super(PFNOptApp, self).__init__(
            description='',
            version=pfnopt.__version__,
            command_manager=command_manager
        )
        for name, cls in _COMMANDS.items():
            command_manager.add_command(name, cls)

    def configure_logging(self):
        # type: () -> None

        super(PFNOptApp, self).configure_logging()

        # Find the StreamHandler that is configured by super's configure_logging,
        # and replace its formatter with our fancy one.
        root_logger = logging.getLogger()
        stream_handlers = [
            handler for handler in root_logger.handlers
            if isinstance(handler, logging.StreamHandler)]
        assert len(stream_handlers) == 1
        stream_handler = stream_handlers[0]
        stream_handler.setFormatter(pfnopt.logging.create_default_formatter())


def main():
    # type: () -> int

    argv = sys.argv[1:] if len(sys.argv) > 1 else ['help']
    return PFNOptApp().run(argv)
