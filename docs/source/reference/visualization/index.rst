.. module:: optuna.visualization

optuna.visualization
====================

The :mod:`~optuna.visualization` module provides utility functions for plotting the optimization process using plotly and matplotlib. Plotting functions take generally take a :class:`~optuna.study.Study` object and optional parameters passed as a list to a ``params`` argument.

.. note::
    In the :mod:`optuna.visualization` module, the following functions use plotly to create figures, but `JupyterLab`_ cannot
    render them by default. Please follow this `installation guide`_ to show figures in
    `JupyterLab`_.

    .. _JupyterLab: https://github.com/jupyterlab/jupyterlab
    .. _installation guide: https://github.com/plotly/plotly.py#jupyterlab-support-python-35

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.visualization.plot_contour
   optuna.visualization.plot_edf
   optuna.visualization.plot_intermediate_values
   optuna.visualization.plot_optimization_history
   optuna.visualization.plot_parallel_coordinate
   optuna.visualization.plot_param_importances
   optuna.visualization.plot_slice
   optuna.visualization.is_available

.. note::
    The following :mod:`optuna.visualization.matplotlib` module uses Matplotlib as a backend.

.. toctree::
    :maxdepth: 1

    matplotlib
