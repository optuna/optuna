.. module:: optuna.visualization


Visualization
=============

.. note::
    :mod:`~optuna.visualization` module uses plotly to create figures, but `JupyterLab`_ cannot
    render them by default. Please follow this `installation guide`_ to show figures in
    `JupyterLab`_.

    .. _JupyterLab: https://github.com/jupyterlab/jupyterlab
    .. _installation guide: https://github.com/plotly/plotly.py#jupyterlab-support-python-35

.. autofunction:: plot_contour

.. autofunction:: plot_intermediate_values

.. autofunction:: plot_optimization_history

.. autofunction:: plot_parallel_coordinate

.. autofunction:: plot_slice

.. autofunction:: is_available
