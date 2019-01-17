"""
Optuna example that optimizes a classifier configuration for Iris dataset using sklearn.

This example is the same as `sklearn_simple.py` except that it uses a callable class for
implementing the objective function. It takes the Iris dataset by a constructor's argument
instead of loading it in each trial execution. This will speed up the execution of each trial
compared to `sklearn_simple.py`.

You can run this example as follows:
    $ python sklearn_additional_args.py

"""

import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm


class Objective(object):
    def __init__(self, iris):
        self.iris = iris

    def __call__(self, trial):
        x, y = self.iris.data, self.iris.target

        classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RandomForest'])
        if classifier_name == 'SVC':
            svc_c = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
            classifier_obj = sklearn.svm.SVC(C=svc_c)
        else:
            rf_max_depth = int(trial.suggest_loguniform('rf_max_depth', 2, 32))
            classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)

        score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
        accuracy = score.mean()
        return 1.0 - accuracy


if __name__ == '__main__':
    import optuna

    # Load the dataset in advance for reusing it each trial execution.
    iris = sklearn.datasets.load_iris()
    objective = Objective(iris)

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print(study.best_trial)
