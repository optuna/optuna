from cliff.app import App
from cliff.command import Command
from cliff.commandmanager import CommandManager
import sys

from pfnopt.study import create_new_study


class PFNOptApp(App):

    def __init__(self):
        super(PFNOptApp, self).__init__(
            description='',
            version='0.0.1',
            command_manager=CommandManager('pfnopt.command')
        )


class MakeStudy(Command):

    def get_parser(self, prog_name):
        parser = super(MakeStudy, self).get_parser(prog_name)

        parser.add_argument('--url', '-u', dest='url', required=True)

    def take_action(self, parsed_args):
        create_new_study()


def main():
    return PFNOptApp().run(sys.argv[1:])


main()
