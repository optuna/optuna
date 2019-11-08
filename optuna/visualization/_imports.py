from optuna import type_checking

try:
    import plotly  # NOQA
    import plotly.graph_objs as go  # NOQA
    from plotly.graph_objs._figure import Figure  # NOQA
    from plotly.subplots import make_subplots  # NOQA
    _available = True
except ImportError as e:
    _import_error = e
    # Visualization features are disabled because plotly is not available.
    _available = False

if type_checking.TYPE_CHECKING:
    from typing import DefaultDict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA

    from plotly.graph_objs import Contour  # NOQA
    from plotly.graph_objs import Scatter  # NOQA

    from optuna.structs import FrozenTrial  # NOQA
