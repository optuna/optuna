"""
.. _first:

Lightweight, versatile, and platform agnostic architecture
==========================================================

Optuna is entirely written in Python and has few dependencies.
This means that we can quickly move to the real example once you get interested in Optuna.


Quadratic Function Example
--------------------------

Usually, Optuna is used to optimize hyperparameters, but as an example,
let's optimize a simple quadratic function: :math:`(x - 2)^2`.
"""


###################################################################################################
# First of all, import :mod:`optuna`.

import optuna


###################################################################################################
# In optuna, conventionally functions to be optimized are named `objective`.


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


###################################################################################################
# This function returns the value of :math:`(x - 2)^2`. Our goal is to find the value of ``x``
# that minimizes the output of the ``objective`` function. This is the "optimization."
# During the optimization, Optuna repeatedly calls and evaluates the objective function with
# different values of ``x``.
#
# A :class:`~optuna.trial.Trial` object corresponds to a single execution of the objective
# function and is internally instantiated upon each invocation of the function.
#
# The `suggest` APIs (for example, :func:`~optuna.trial.Trial.suggest_float`) are called
# inside the objective function to obtain parameters for a trial.
# :func:`~optuna.trial.Trial.suggest_float` selects parameters uniformly within the range
# provided. In our example, from :math:`-10` to :math:`10`.
#
# To start the optimization, we create a study object and pass the objective function to method
# :func:`~optuna.study.Study.optimize` as follows.

study = optuna.create_study()
study.optimize(objective, n_trials=100)


###################################################################################################
# You can get the best parameter as follows.

best_params = study.best_params
found_x = best_params["x"]
print("Found x: {}, (x - 2)^2: {}".format(found_x, (found_x - 2) ** 2))

###################################################################################################
# We can see that the ``x`` value found by Optuna is close to the optimal value of ``2``.

###################################################################################################
# .. note::
#     When used to search for hyperparameters in machine learning,
#     usually the objective function would return the loss or accuracy
#     of the model.


###################################################################################################
# Study Object
# ------------
#
# Let us clarify the terminology in Optuna as follows:
#
# * **Trial**: A single call of the objective function
# * **Study**: An optimization session, which is a set of trials
# * **Parameter**: A variable whose value is to be optimized, such as ``x`` in the above example
#
# In Optuna, we use the study object to manage optimization.
# Method :func:`~optuna.study.create_study` returns a study object.
# A study object has useful properties for analyzing the optimization outcome.

###################################################################################################
# To get the dictionary of parameter name and parameter values:


study.best_params

###################################################################################################
# To get the best observed value of the objective function:

study.best_value


###################################################################################################
# To get the best trial:

study.best_trial


###################################################################################################
# To get all trials:

study.trials
for trial in study.trials[:2]:  # Show first two trials
    print(trial)

###################################################################################################
# To get the number of trials:

len(study.trials)


###################################################################################################
# By executing :func:`~optuna.study.Study.optimize` again, we can continue the optimization.

study.optimize(objective, n_trials=100)


###################################################################################################
# To get the updated number of trials:

len(study.trials)


###################################################################################################
# As the objective function is so easy that the last 100 trials don't improve the result.
# However, we can check the result again:
best_params = study.best_params
found_x = best_params["x"]
print("Found x: {}, (x - 2)^2: {}".format(found_x, (found_x - 2) ** 2))
