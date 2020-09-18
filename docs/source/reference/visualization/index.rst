optuna.visualization
====================

.. note::
    In the :mod:`~optuna.visualization` module, the following functions uses plotly to create figures, but `JupyterLab`_ cannot
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
    The following :mod:`~optuna.visualization.matplotlib` module uses matplotlib as a backend.

.. toctree::
    :maxdepth: 1

    matplotlib
