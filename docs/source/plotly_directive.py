"""
Usage:
```
.. plotly::

    import plotly.graph_objects as go
    import numpy as np

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    go.Figure(data=go.Scatter(x=t, y=y, mode='markers'))
```

Requirements:
1. docstring contains a single code block.
2. the code block ends with an expression that evaluates to a plotly figure.

Based on:
https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/sphinxext/plot_directive.py
"""

import os
import textwrap
import traceback

from docutils.parsers.rst import Directive
import jinja2  # Sphinx dependency.

import plotly


class PlotlyDirective(Directive):
    """The ``.. plotly::`` directive, as documented in the module's docstring."""

    has_content = True

    def run(self):
        """Run the plotly directive."""
        try:
            return run(
                self.arguments,
                self.content,
                self.options,
                self.state_machine,
                self.state,
                self.lineno,
            )
        except Exception as e:
            raise self.error(str(e))


def setup(app):
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir
    app.add_directive("plotly", PlotlyDirective)

    metadata = {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
        "version": plotly.__version__,
    }
    return metadata


RST_TEMPLATE = """
.. code-block:: python

{{ source_code }}

.. raw:: html

   <iframe src="./{{ figure_name }}"
     width="100%" height="500px" frameborder="0">
   </iframe>
"""


class PlotError(RuntimeError):
    pass


def out_of_date(original, derived):
    return os.stat(derived).st_mtime < os.stat(original).st_mtime


def save_plotly_figure(fig, path):
    fig_html = plotly.offline.plot(fig, output_type="div", include_plotlyjs="cdn", auto_open=False)
    with open(path, "w") as f:
        f.write(fig_html)


def assign_last_line_into_variable(code, variable_name):
    *rest, last = code.strip().split("\n")
    last = "{} = ".format(variable_name) + last
    return "\n".join([*rest, last])


def run_code(code):
    namespace = {}
    variable_name = "fig"

    try:
        exec(assign_last_line_into_variable(code, variable_name), namespace)
    except (Exception, SystemExit) as err:
        raise PlotError(traceback.format_exc()) from err

    return namespace[variable_name]


def get_fig_out_path(rst_path, conf_dir, builder_out_dir):
    rst_name = os.path.basename(rst_path)
    base = os.path.splitext(rst_name)[0]
    fig_out_name = base.replace(".", "-") + ".html"
    out_dir = os.path.dirname(os.path.relpath(rst_path, start=conf_dir))
    out_dir = os.path.join(builder_out_dir, out_dir)
    return os.path.join(out_dir, fig_out_name)


def run(arguments, content, options, state_machine, state, lineno):
    rst_path = state_machine.document.attributes["source"]
    fig_out_path = get_fig_out_path(rst_path, setup.confdir, setup.app.builder.outdir)
    os.makedirs(os.path.dirname(fig_out_path), exist_ok=True)

    # run code and save generated plotly figure
    try:
        code = textwrap.dedent("\n".join(map(str, content)))
        if not os.path.exists(fig_out_path) or out_of_date(rst_path, fig_out_path):
            fig = run_code(code)
            save_plotly_figure(fig, fig_out_path)

        errors = []
    except PlotError as err:
        reporter = state.memo.reporter
        sm = reporter.system_message(
            2,
            "Exception occurred in plotting {}:\n{}".format(rst_path, err),
            line=lineno,
        )
        errors = [sm]

    # generate output restructuredtext
    rst = jinja2.Template(RST_TEMPLATE).render(
        source_code=textwrap.indent(code, " " * 4),
        figure_name=os.path.basename(fig_out_path),
    )
    total_lines = [*rst.split("\n"), "\n"]
    state_machine.insert_input(total_lines, source=rst_path)

    return errors
