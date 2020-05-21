"""
Optuna example that optimizes a configuration for Olivetti faces dataset using
skimage.

In this example, we optimize a classifier configuration for Olivetti faces dataset.
We optimize parameters of local_binary_pattern function in skimage and the
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

import optuna


def load_data():
    rng = np.random.RandomState(0)
    dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
    faces = dataset.images
    target = dataset.target
    classes = np.unique(target)
    classes.sort()

    ref_index = np.argmax(target == classes[:, None], axis=1)
    valid_index = np.delete(np.arange(len(faces)), ref_index)

    x_ref = faces[ref_index]
    y_ref = target[ref_index]
    x_valid = faces[valid_index]
    y_valid = target[valid_index]
    return x_ref, x_valid, y_ref, y_valid


def get_lbp_hist(img, P, R, method):
    lbp = ft.local_binary_pattern(img, P, R, method)
    if method == "uniform":
        bin_max = P + 3
        range_max = P + 2
    elif method == "default":
        bin_max = 2 ** P
        range_max = 2 ** P - 1
    hist, _ = np.histogram(
        lbp.ravel(), density=True, bins=np.arange(0, bin_max), range=(0, range_max)
    )
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
    if metric == "kl":
        dist = calc_kl_dist(p, q)
    elif metric == "cos":
        dist = calc_cos_dist(p, q)
    elif metric == "euc":
        dist = calc_euc_dist(p, q)
    return dist


def objective(trial):
    # Get Olivetti faces dataset.
    x_ref, x_valid, y_ref, y_valid = load_data()

    # We optimize parameters of local_binary_pattern function in skimage
    # and the choice of distance metric classes.
    P = trial.suggest_int("P", 1, 15)
    R = trial.suggest_uniform("R", 1, 10)
    method = trial.suggest_categorical("method", ["default", "uniform"])
    metric = trial.suggest_categorical("metric", ["kl", "cos", "euc"])

    x_ref_hist = img2hist(x_ref, P, R, method)
    x_valid_hist = img2hist(x_valid, P, R, method)
    dist = calc_dist(x_ref_hist, x_valid_hist, metric)

    y_pred = np.argmin(dist, axis=1)
    accuracy = sklearn.metrics.accuracy_score(y_valid, y_pred)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=600)
    print(study.best_trial)
