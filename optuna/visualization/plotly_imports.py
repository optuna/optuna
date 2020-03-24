try:
    import plotly  # NOQA
    import plotly.graph_objs as go  # NOQA
    from plotly.graph_objs import Contour, Scatter  # NOQA
    from plotly.subplots import make_subplots  # NOQA

    _available = True
except ImportError as e:
    _import_error = e
    # Visualization features are disabled because plotly is not available.
    _available = False
