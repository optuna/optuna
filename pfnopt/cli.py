from __future__ import absolute_import

from argparse import ArgumentParser  # NOQA
from argparse import Namespace  # NOQA
from cliff.app import App
from cliff.command import Command
from cliff.commandmanager import CommandManager
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


_COMMANDS = {
    'create-study': CreateStudy,
    'study set-user-attr': StudySetUserAttribute,
    'dashboard': Dashboard,
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
