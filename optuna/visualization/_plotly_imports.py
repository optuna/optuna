from distutils.version import StrictVersion

from optuna._imports import try_import


with try_import() as _imports:  # NOQA
    import plotly  # NOQA
    import plotly.graph_objs as go  # NOQA
    from plotly.graph_objs import Contour, Scatter  # NOQA
    from plotly.subplots import make_subplots  # NOQA

    from plotly import __version__ as plotly_version

    if StrictVersion(plotly_version) < StrictVersion("4.0.0"):
        raise ImportError(
            "Your version of Plotly is " + plotly_version + " . "
            "Please install plotly version 4.0.0 or higher. "
            "Plotly can be installed by executing `$ pip install -U plotly>=4.0.0`. "
            "For further information, please refer to the installation guide of plotly. ",
            name="plotly",
        )
