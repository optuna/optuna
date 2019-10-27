"""
Optuna distributed optimization example that optimizes an sklearn classifier configuration for Iris dataset on Kubernetes.

This example's code is mostly the same as the sklearn_simple.py example, except for two things:

1 - It gives a name to the study and sets load_if_exists to True
in order to avoid errors when the code is run from multiple workers.

2 - It sets the storage address to the postgres pod deployed with the workers.

"""
import os
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RandomForest'])
    if classifier_name == 'SVC':
        svc_c = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma='auto')
    else:
        rf_max_depth = int(trial.suggest_loguniform('rf_max_depth', 2, 32))
        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=10)

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy


if __name__ == '__main__':
    import optuna
    study = optuna.create_study(
        direction='maximize',
        study_name='kubernetes',
        storage='postgresql://{}:{}@postgres:5432/{}'.format(
            os.environ['POSTGRES_USER'],
            os.environ['POSTGRES_PASSWORD'],
            os.environ['POSTGRES_DB'],
        ),
        load_if_exists=True
    )
    study.optimize(objective, n_trials=20)
    print(study.best_trial)
