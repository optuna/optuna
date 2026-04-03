Installation
============

Optuna supports Python 3.9 or newer.

We recommend to install Optuna via pip:

.. code-block:: bash

    $ pip install optuna

You can also install the development version of Optuna from master branch of Git repository:

.. code-block:: bash

    $ pip install git+https://github.com/optuna/optuna.git

You can also install Optuna via conda:

.. code-block:: bash

    $ conda install -c conda-forge optuna


Optional: Accelerate with numba
--------------------------------

Optuna can optionally use `numba <https://numba.pydata.org/>`__ to JIT-compile
performance-critical numerical code paths in the samplers. When numba is installed,
functions such as the error function, log-normal CDF, Pareto front detection, and
hypervolume computation are automatically accelerated. When numba is not installed,
Optuna transparently falls back to its pure-Python/NumPy implementations.

To install Optuna with numba support:

.. code-block:: bash

    $ pip install optuna[numba]

This is most beneficial for studies with many trials (500+) and cheap objective
functions, where the sampler's suggestion overhead becomes a significant fraction
of the total runtime.
