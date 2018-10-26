.. _attributes:

User Attributes
===============

This feature is to annotate experiments with user-defined attributes.


Adding User Attributes to Studies
---------------------------------

A ``Study`` object provides ``set_user_attr`` method to register a pair of key and value as an user-defined attribute.
A key is supposed to be a ``str``, and a value be any object serializable with ``json.dumps``.

.. code-block:: python

    study = ...  # type: optuna.Study
    study.set_user_attr('contributors', ['Akiba', 'Sano'])
    study.set_user_attr('dataset', 'MNIST')


We can access annotated attributes with ``user_attrs`` property.

.. code-block:: python

    study.user_attrs  # {'contributors': ['Akiba', 'Sano'], 'dataset': 'MNIST'}

``get_all_study_summaries`` method also collects user-defined attributes for each study.

.. code-block:: python

    study_summaries = optuna.get_all_study_summaries(storage_url)
    study_summaries[0].user_attrs  # {'contributors': ['Akiba', 'Sano'], 'dataset': 'MNIST'}

Note: see also ``optuna study set-user-attr`` command, which set an attribute via command line interface.


Adding User Attributes to Trials
--------------------------------

As with ``Study``, a ``Trial`` object provides ``set_user_attr`` method.
Attributes are set inside an objective function.

.. code-block:: python

    def objective(trial):
        trial.set_user_attr('task', 'quadratic function')

        x = trial.suggest_uniform('x', -10, 10)
        return (x - 2) ** 2


We can access annotated attributes as:

.. code-block:: python

    study.trials[0].user_attrs['task']  # 'quadratic function'

Note that, in this example, the attribute is not annotated to a ``Study`` but a single ``Trial``.
