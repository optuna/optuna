|optunalogo|

Optuna: A hyperparameter optimization framework
===============================================

*Optuna* is an automatic hyperparameter optimization software framework,
particularly designed for machine learning. It features an imperative,
*define-by-run* style user API. Thanks to our *define-by-run* API, the
code written with Optuna enjoys high modularity, and the user of Optuna
can dynamically construct the search spaces for the hyperparameters.

Key Features
------------

Optuna has modern functionalities as follows:

- :doc:`Lightweight, versatile, and platform agnostic architecture <tutorial/10_key_features/001_first>`

  - Handle a wide variety of tasks with a simple installation that has few requirements.

- :doc:`Pythonic search spaces <tutorial/10_key_features/002_configurations>`

  - Define search spaces using familiar Python syntax including conditionals and loops.

- :doc:`Efficient optimization algorithms <tutorial/10_key_features/003_efficient_optimization_algorithms>`

  - Adopt state-of-the-art algorithms for sampling hyperparameters and efficiently pruning unpromising trials.

- :doc:`Easy parallelization <tutorial/10_key_features/004_distributed>`

  - Scale studies to tens or hundreds or workers with little or no changes to the code.

- :doc:`Quick visualization <tutorial/10_key_features/005_visualization>`

  - Inspect optimization histories from a variety of plotting functions.

Basic Concepts
--------------

We use the terms *study* and *trial* as follows:

-  Study: optimization based on an objective function
-  Trial: a single execution of the objective function

Please refer to sample code below. The goal of a *study* is to find out
the optimal set of hyperparameter values (e.g., ``classifier`` and
``svm_c``) through multiple *trials* (e.g., ``n_trials=100``). Optuna is
a framework designed for the automation and the acceleration of the
optimization *studies*.

|Open in Colab|

.. code:: python

    import ...

    # Define an objective function to be minimized.
    def objective(trial):

        # Invoke suggest methods of a Trial object to generate hyperparameters.
        regressor_name = trial.suggest_categorical('classifier', ['SVR', 'RandomForest'])
        if regressor_name == 'SVR':
            svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
            regressor_obj = sklearn.svm.SVR(C=svr_c)
        else:
            rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
            regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth)

        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state=0)

        regressor_obj.fit(X_train, y_train)
        y_pred = regressor_obj.predict(X_val)

        error = sklearn.metrics.mean_squared_error(y_val, y_pred)

        return error  # An objective value linked with the Trial object.

    study = optuna.create_study()  # Create a new study.
    study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.

Communication
-------------

-  `GitHub Discussions <https://github.com/optuna/optuna/discussions>`__ for questions.
-  `GitHub Issues <https://github.com/optuna/optuna/issues>`__ for bug
   reports and feature requests.

Contribution
------------

Any contributions to Optuna are welcome! When you send a pull request,
please follow the `contribution guide <https://github.com/optuna/optuna/blob/master/CONTRIBUTING.md>`__.

License
-------

MIT License (see `LICENSE <https://github.com/optuna/optuna/blob/master/LICENSE>`__).

Optuna uses the codes from SciPy and fdlibm projects (see :doc:`Third-party License <license_thirdparty>`).

Reference
---------

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori
Koyama. 2019. Optuna: A Next-generation Hyperparameter Optimization
Framework. In KDD (`arXiv <https://arxiv.org/abs/1907.10902>`__).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorial/index
   reference/index
   faq

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |optunalogo| image:: https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png
  :width: 800
  :alt: OPTUNA
.. |Open in Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: http://colab.research.google.com/github/optuna/optuna-examples/blob/main/quickstart.ipynb
