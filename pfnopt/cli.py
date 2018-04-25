from __future__ import absolute_import

from argparse import ArgumentParser  # NOQA
from argparse import Namespace  # NOQA
from cliff.app import App
from cliff.command import Command
from cliff.commandmanager import CommandManager
import imp
import logging
import sys
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import List  # NOQA

import pfnopt


class BaseCommand(Command):

    def __init__(self, *args, **kwargs):
        # type: (List[Any], Dict[str, Any]) -> None

        super(BaseCommand, self).__init__(*args, **kwargs)
        self.logger = pfnopt.logging.get_logger(__name__)


class CreateStudy(BaseCommand):

    def get_parser(self, prog_name):
        # type: (str) -> ArgumentParser

        parser = super(CreateStudy, self).get_parser(prog_name)
        parser.add_argument('--url', '-u', required=True)
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> None

        storage = pfnopt.storages.RDBStorage(parsed_args.url)
        study_uuid = pfnopt.create_study(storage).study_uuid
        print(study_uuid)


class StudySetUserAttribute(BaseCommand):

    def get_parser(self, prog_name):
        # type: (str) -> ArgumentParser

        parser = super(StudySetUserAttribute, self).get_parser(prog_name)
        parser.add_argument('--url', '-u', required=True)
        parser.add_argument('--study-uuid', required=True)
        parser.add_argument('--key', '-k', required=True)
        parser.add_argument('--value', '-v', required=True)
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> None

        study = pfnopt.Study(storage=parsed_args.url, study_uuid=parsed_args.study_uuid)
        study.set_user_attr(parsed_args.key, parsed_args.value)

        self.logger.info('Attribute successfully written.')


class Dashboard(BaseCommand):

    def get_parser(self, prog_name):
        # type: (str) -> ArgumentParser

        parser = super(Dashboard, self).get_parser(prog_name)
        parser.add_argument('--url', '-u', required=True)
        parser.add_argument('--study-uuid', required=True)
        parser.add_argument('--out', '-o')
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> None

        study = pfnopt.Study(storage=parsed_args.url, study_uuid=parsed_args.study_uuid)

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
        parser.add_argument('--timeout-seconds', type=float)
        parser.add_argument('--n-jobs', type=int, default=1)
        parser.add_argument('--url', '-u')
        parser.add_argument('--study-uuid')
        parser.add_argument('--create-study', action='store_true')
        parser.add_argument('file')
        parser.add_argument('method')
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> int

        if parsed_args.create_study and parsed_args.study_uuid:
            raise ValueError('Inconsistent arguments. Flags --create-study and --study-uuid '
                             'should not be specified at the same time.')
        if not parsed_args.create_study and not parsed_args.study_uuid:
            raise ValueError('Inconsistent arguments. Either --create-study or --study-uuid '
                             'should be specified.')

        if parsed_args.create_study:
            study = pfnopt.create_study(storage=parsed_args.url)
        else:
            study = pfnopt.Study(storage=parsed_args.url, study_uuid=parsed_args.study_uuid)

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
            timeout_seconds=parsed_args.timeout_seconds, n_jobs=parsed_args.n_jobs,
            study=study)
        return 0


_COMMANDS = {
    'create-study': CreateStudy,
    'study set-user-attr': StudySetUserAttribute,
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
