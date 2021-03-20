import os
from sklearn.datasets import fetch_openml

import numpy as np
import torch


CHAINER_ROOT = "pfnet/chainer/mnist"
PYTORCH_ROOT = "MNIST/processed"


if __name__ == "__main__":

    custom_data_home = os.getcwd()
    print(custom_data_home)
    X, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False, data_home=custom_data_home)
    y = y.astype('f')

    chainer_root = os.path.join(custom_data_home, CHAINER_ROOT)
    pytorch_root = os.path.join(custom_data_home, PYTORCH_ROOT)

    if not os.path.exists(chainer_root):
        os.makedirs(chainer_root)

    if not os.path.exists(pytorch_root):
        os.makedirs(pytorch_root)

    X = X.reshape(-1, 28, 28)
    y = y[:, None]
    trn_X, test_X = X[:-10000], X[-10000:]
    trn_y, test_y = y[:-10000], y[-10000:]

    torch.save(
        (torch.from_numpy(trn_X), torch.from_numpy(trn_y)),
        os.path.join(pytorch_root, "training.pt"),
    )
    torch.save(
        (torch.from_numpy(test_X), torch.from_numpy(test_y)),
        os.path.join(pytorch_root, "test.pt")
    )

    np.savez(
        os.path.join(custom_data_home, os.path.join(chainer_root, "train.npz")),
        x=trn_X,
        y=trn_y.ravel(),
    )
    np.savez(
        os.path.join(custom_data_home, os.path.join(chainer_root, "test.npz")),
        x=test_X,
        y=test_y.ravel(),
    )
