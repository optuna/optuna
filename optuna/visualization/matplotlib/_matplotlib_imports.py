from packaging import version

from optuna._imports import try_import


with try_import() as _imports:  # NOQA
    # TODO(ytknzw): Add specific imports.
    import matplotlib  # NOQA
    from matplotlib import __version__ as matplotlib_version
    from matplotlib import cm  # NOQA
    from matplotlib import pyplot as plt  # NOQA
    from matplotlib.axes._axes import Axes  # NOQA
    from matplotlib.collections import LineCollection  # NOQA
    from matplotlib.collections import PathCollection  # NOQA
    from matplotlib.colors import Colormap  # NOQA
    from matplotlib.contour import ContourSet  # NOQA
    from matplotlib.patches import Rectangle  # NOQA

    # TODO(ytknzw): Set precise version.
    if version.parse(matplotlib_version) < version.parse("3.0.0"):
        raise ImportError(
            "Your version of Matplotlib is " + matplotlib_version + " . "
            "Please install Matplotlib version 3.0.0 or higher. "
            "Matplotlib can be installed by executing `$ pip install -U matplotlib>=3.0.0`. "
            "For further information, please refer to the installation guide of Matplotlib. ",
            name="matplotlib",
        )
