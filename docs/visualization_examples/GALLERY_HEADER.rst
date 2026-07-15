.. _visualization-examples-index:

.. _general_visualization_examples:

optuna.visualization
====================

The :mod:`~optuna.visualization` module provides utility functions for plotting the optimization process using plotly and matplotlib. Plotting functions generally take a :class:`~optuna.study.Study` object and optional parameters are passed as a list to the ``params`` argument.

.. note::
    In the :mod:`optuna.visualization` module, the following functions use plotly to create figures, but `JupyterLab`_ cannot
    render them by default. Please follow this `installation guide`_ to show figures in
    `JupyterLab`_.

    .. _JupyterLab: https://github.com/jupyterlab/jupyterlab
    .. _installation guide: https://github.com/plotly/plotly.py#jupyterlab-support
