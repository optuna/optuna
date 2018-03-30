from __future__ import absolute_import

from argparse import ArgumentParser  # NOQA
from argparse import Namespace  # NOQA
from cliff.app import App
from cliff.command import Command
from cliff.commandmanager import CommandManager
import logging
import sys

import pfnopt


class MakeStudy(Command):

    def get_parser(self, prog_name):
        # type: (str) -> ArgumentParser

        parser = super(MakeStudy, self).get_parser(prog_name)
        parser.add_argument('--url', '-u', dest='url', required=True)
        return parser

    def take_action(self, parsed_args):
        # type: (Namespace) -> None

        storage = pfnopt.storages.RDBStorage(parsed_args.url)
        study_uuid = pfnopt.create_study(storage).study_uuid
        print(study_uuid)


_COMMANDS = {
    'mkstudy': MakeStudy
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
