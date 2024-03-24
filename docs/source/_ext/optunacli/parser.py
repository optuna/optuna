from __future__ import annotations

from argparse import ArgumentParser

from optuna.cli import _get_parser


def _format_usage(parser: ArgumentParser) -> str:
    """Format the usage without any prefixes"""
    fmt = parser._get_formatter()
    fmt.add_usage(parser.usage, parser._actions, parser._mutually_exclusive_groups, prefix="")
    return fmt.format_help().strip()


def parse_arguments(parser, shared_actions: set | None = None):
    """Collect all arguments"""
    shared_actions = shared_actions or set()
    ignored_actions = {"==SUPPRESS==", "help"}

    action_groups = []
    for action_group in parser._action_groups:
        actions = []
        for action in action_group._group_actions:
            # Skip arguments shared among all subcommands.
            if action.dest in shared_actions or action.dest in ignored_actions:
                continue
            action_data = {
                "name": action.dest,
                "default": f'"{action.default}"',
                "help": action.help,
                "choices": action.choices,
                "prog": action.option_strings,
            }
            actions.append(action_data)
        # The titles are either 'positional arguments', 'options' or 'shared_actions'
        # until any custom action group is added.
        if actions:
            action_groups.append({"title": action_group.title, "actions": actions})
    return action_groups


def parse_parser(parser: ArgumentParser, shared_actions: set[str]) -> dict:
    """Parse an ArgumentParser object into a dict."""
    data = {
        "name": parser.prog,
        "description": parser.description,
        "usage": _format_usage(parser),
        "prog": parser.prog,
        "action_groups": parse_arguments(parser, shared_actions=shared_actions),
    }
    return data


def parse_parsers():
    "Collect the shared optional arguments among all subcommands."
    main_parser, parent_parser, command_name_to_subparser = _get_parser()

    # Currently, only unnamed optional arguments are shared among all subcommands.
    shared_actions = parse_arguments(parent_parser)[0]
    shared_actions["title"] = "shared options"
    shared_actions_set = {action["name"] for action in shared_actions["actions"]}

    main_parser.prog = "optuna"
    parsed_args = parse_parser(main_parser, shared_actions_set)
    parsed_args["action_groups"].append(shared_actions)

    parsed_args["subcommands"] = []
    for command_name, subparser in command_name_to_subparser.items():
        subparser.prog = f"optuna {command_name}"
        parsed_args["subcommands"].append(parse_parser(subparser, shared_actions_set))

    return parsed_args
