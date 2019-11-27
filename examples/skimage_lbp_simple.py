"""
Optuna example that optimizes a configuration for Olivetti faces dataset using
skimage.

In this example, we optimize a classifier configuration for Olivetti faces dataset.
We optimize paraterers of local_binary_pattern function in skimage and the
choice of distance metric classes.

We have following two ways to execute this example:

(1) Execute this code directly.
    $ python skimage_lbp_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize skimage_lbp_simple.py objective --n-trials=100 --study \
      $STUDY_NAME --storage sqlite:///example.db

"""

import numpy as np
import skimage.feature as ft
from sklearn.datasets import fetch_olivetti_faces
import sklearn.metrics


def load_data():
    rng = np.random.RandomState(0)
    dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
    faces = dataset.images
    target = dataset.target
    classes = sorted(list(set(target)))
    ref_index = [(np.min(np.where(target == index))) for index in classes]
    x_ref = faces[ref_index]
    y_ref = target[ref_index]
    test_index = [index for index in range(len(faces)) if index not in ref_index]
    x_test = faces[test_index]
    y_test = target[test_index]
    return x_ref, x_test, y_ref, y_test


def get_lbp_hist(img, P, R, method):
    lbp = ft.local_binary_pattern(img, P, R, method)
    if method == 'uniform':
        hist, _ = np.histogram(lbp.ravel(), density=True,
                               bins=np.arange(0, P + 3),
                               range=(0, P + 2))
    elif method == 'default':
        hist, _ = np.histogram(lbp.ravel(), density=True,
                               bins=np.arange(0, 2**P),
                               range=(0, 2**P-1))
    return hist


def img2hist(imgs, P, R, method):
    hists = [get_lbp_hist(img, P, R, method) for img in imgs]
    return np.array(hists)


def log2matrix(p):
    return np.where(p != 0, np.log2(p), 0)


def calc_kl_dist(p, q):
    dist = np.matmul(p * log2matrix(p), np.where(q != 0, 1, 0).T) - np.matmul(p, log2matrix(q).T)
    return dist.T


def calc_euc_dist(p, q):
    p_norm = np.diag(np.dot(p, p.T))
    q_norm = np.vstack(np.diag(np.dot(q, q.T)))
    dist = np.sqrt(-2 * np.matmul(q, p.T) + p_norm + q_norm)
    return dist


def calc_cos_dist(p, q):
    p_norm = np.diag(np.dot(p, p.T))
    q_norm = np.vstack(np.diag(np.dot(q, q.T)))
    dist = 1 - np.matmul(q, p.T) / (np.sqrt(p_norm) * np.sqrt(q_norm))
    return dist


def calc_dist(p, q, metric):
    if metric == 'kl':
        dist = calc_kl_dist(p, q)
    elif metric == 'cos':
        dist = calc_cos_dist(p, q)
    elif metric == 'euc':
        dist = calc_euc_dist(p, q)
    return dist


def objective(trial):
    # Get Olivetti faces dataset.
    x_ref, x_test, y_ref, y_test = load_data()

    # We optimzie paraterers of local_binary_pattern function in skimage
    # and the choice of distance metric classes.
    P = trial.suggest_int('P', 1, 15)
    R = trial.suggest_uniform('R', 1, 10)
    method = trial.suggest_categorical('method', ['default', 'uniform'])
    metric = trial.suggest_categorical('metric', ['kl', 'cos', 'euc'])

    x_ref_hist = img2hist(x_ref, P, R, method)
    x_test_hist = img2hist(x_test, P, R, method)
    dist = calc_dist(x_ref_hist, x_test_hist, metric)

    y_pred = np.argmin(dist, axis=1)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    return accuracy


if __name__ == '__main__':
    import optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    print(study.best_trial)
