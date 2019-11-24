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


def getLocalBinaryPatternhist(img, P, R, method):
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


def gethists(imgs, P, R, method):
    hists = [getLocalBinaryPatternhist(img, P, R, method) for img in imgs]
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


def calc_dist(p, q, measure):
    if measure == 'kl':
        dist = calc_kl_dist(p, q)
    elif measure == 'cos':
        dist = calc_cos_dist(p, q)
    elif measure == 'euc':
        dist = calc_euc_dist(p, q)
    return dist


def objective(trial):
    P = trial.suggest_int('P', 1, 15)
    R = trial.suggest_uniform('R', 1, 10)
    method = trial.suggest_categorical('method', ['default', 'uniform'])
    measure = trial.suggest_categorical('measure', ['kl', 'cos', 'euc'])
    x_ref, x_test, y_ref, y_test = load_data()
    x_ref_hist = gethists(x_ref, P, R, method)
    x_test_hist = gethists(x_test, P, R, method)
    dist = calc_dist(x_ref_hist, x_test_hist, measure)
    y_pred = np.argmin(dist, axis=1)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    return accuracy


if __name__ == '__main__':
    import optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    print(study.best_trial)
