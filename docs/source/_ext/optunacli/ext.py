from __future__ import annotations

from typing import Any

from docutils import nodes
from docutils.frontend import OptionParser
from docutils.parsers.rst import Directive
from docutils.parsers.rst import Parser
from docutils.parsers.rst.directives import unchanged
from docutils.statemachine import StringList
from docutils.utils import new_document
import sphinx

from optuna.version import __version__

from .parser import parse_parsers


def render_list(
    structured_texts: list[str | nodes.definition], settings: Any = None
) -> list[nodes.Node]:
    """
    Given a list of reStructuredText or MarkDown sections, return a docutils node list
    """
    if len(structured_texts) == 0:
        return []
    all_subcommands = []
    for element in structured_texts:
        if isinstance(element, str):
            if settings is None:
                settings = OptionParser(components=(Parser,)).get_default_values()
            document = new_document(None, settings)
            Parser().parse(element + "\n", document)
            all_subcommands += document.children
        elif isinstance(element, nodes.definition):
            all_subcommands += element

    return all_subcommands


def print_action_groups(data: dict[str, Any], settings: Any = None) -> list[nodes.section]:
    nodes_list = []
    for action_group in data["action_groups"]:
        # Create a new section for each action group, e.g., positional arguments or options.
        section = nodes.section(ids=[action_group["title"].replace(" ", "-").lower()])
        section += nodes.title(action_group["title"], action_group["title"])

        # Collect all arguments in the action group except for the ones shared among all
        # subcommands.
        arguments = []
        for action in action_group["actions"]:
            argument = []
            if action.get("help"):
                argument.append(action["help"])
            if action.get("description"):
                argument.append(action["description"])

            # Add possible choices and default values as a field list.
            if action.get("choices"):
                argument.append(
                    f"Possible choices: {', '.join(str(choice) for choice in action['choices'])}\n"
                )
            default = action.get("default")
            if default not in ['"==SUPPRESS=="', "==SUPPRESS==", None]:
                argument.append(f"Default: {default}")

            term = ", ".join(action["prog"])
            arguments.append(
                nodes.option_list_item(
                    "",
                    nodes.option_group("", nodes.option_string(text=term)),
                    nodes.description("", *render_list(argument, settings)),
                )
            )

        section += nodes.option_list("", *arguments)
        nodes_list.append(section)
    return nodes_list


def create_section(action: dict[str, Any], title: str | None = None, settings: Any = None):
    # Create a new section for each action.
    title = title or action["name"]
    sections = nodes.section(ids=title)
    sections += nodes.title(title, title)
    if action.get("help"):
        sections += nodes.paragraph(text=action["help"])

    sections += nodes.literal_block(text=action["usage"])
    for section in print_action_groups(action, settings=settings):
        sections += section

    return sections


class ArgParseDirective(Directive):
    has_content = True
    option_spec = dict(
        deprecated=unchanged,
    )

    def _nested_parse_paragraph(self, text: str) -> nodes.paragraph:
        content = nodes.paragraph()
        self.state.nested_parse(StringList(text.split("\n")), 0, content)
        return content

    def run(self) -> list[nodes.Node]:
        # Set deprecated subcommand if specified.
        self.deprecated = self.options.get("deprecated") or []

        parsed_args = parse_parsers()

        # Add common contents to the document.
        items = []
        items.extend(
            create_section(
                parsed_args,
                settings=self.state.document.settings,
            )
        )
        for subcommand in parsed_args["subcommands"]:
            if subcommand["name"] in self.deprecated:
                continue
            items.extend(
                create_section(
                    subcommand,
                    settings=self.state.document.settings,
                )
            )
        return items


def setup(app: sphinx.application.Sphinx) -> dict[str, bool | str]:
    app.add_directive("argparse", ArgParseDirective)
    return {"parallel_read_safe": True, "version": __version__}
