.. _attributes:

User Attributes
===============

This feature is to annotate experiments with user-defined attributes.


Adding User Attributes to Studies
---------------------------------

A :class:`~optuna.study.Study` object provides :func:`~optuna.study.Study.set_user_attr` method to
register a pair of key and value as an user-defined attribute.
A key is supposed to be a ``str``, and a value be any object serializable with ``json.dumps``.

.. code-block:: python

    import optuna
    study = optuna.create_study(storage='sqlite:///example.db')
    study.set_user_attr('contributors', ['Akiba', 'Sano'])
    study.set_user_attr('dataset', 'MNIST')


We can access annotated attributes with :attr:`~optuna.study.Study.user_attr` property.

.. code-block:: python

    study.user_attrs  # {'contributors': ['Akiba', 'Sano'], 'dataset': 'MNIST'}

:class:`~optuna.struct.StudySummary` object, which can be retrieved by
:func:`~optuna.study.get_all_study_summaries`, also contains user-defined attributes.

.. code-block:: python

    study_summaries = optuna.get_all_study_summaries('sqlite:///example.db')
    study_summaries[0].user_attrs  # {'contributors': ['Akiba', 'Sano'], 'dataset': 'MNIST'}

.. seealso::
    ``optuna study set-user-attr`` command, which sets an attribute via command line interface.


Adding User Attributes to Trials
--------------------------------

As with :class:`~optuna.study.Study`, a :class:`~optuna.trial.Trial` object provides
:func:`~optuna.trial.Trial.set_user_attr` method.
Attributes are set inside an objective function.

.. code-block:: python

    def objective(trial):
        iris = sklearn.datasets.load_iris()
        x, y = iris.data, iris.target

        svc_c = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
        clf = sklearn.svm.SVC(C=svc_c)
        accuracy = sklearn.model_selection.cross_val_score(clf, x, y).mean()

        trial.set_user_attr('accuracy', accuracy)

        return 1.0 - accuracy  # return error for minimization


We can access annotated attributes as:

.. code-block:: python

    study.trials[0].user_attrs  # {'accuracy': 0.83}

Note that, in this example, the attribute is not annotated to a :class:`~optuna.study.Study`
but a single :class:`~optuna.trial.Trial`.
