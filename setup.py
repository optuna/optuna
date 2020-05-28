import os
import sys

import pkg_resources
from setuptools import find_packages
from setuptools import setup

from typing import Dict
from typing import List
from typing import Optional


def get_version() -> str:

    version_filepath = os.path.join(os.path.dirname(__file__), "optuna", "version.py")
    with open(version_filepath) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]
    assert False


def get_long_description() -> str:

    readme_filepath = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_filepath) as f:
        return f.read()


def get_install_requires() -> List[str]:

    return [
        "alembic",
        "cliff",
        "cmaes>=0.5.0",
        "colorlog",
        "joblib",
        "numpy",
        "scipy!=1.4.0",
        "sqlalchemy>=1.1.0",
        "tqdm",
    ]


def get_tests_require() -> List[str]:

    return get_extras_require()["testing"]


def get_extras_require() -> Dict[str, List[str]]:

    requirements = {
        "checking": ["black", "hacking", "mypy"],
        "codecov": ["codecov", "pytest-cov"],
        "doctest": [
            "cma",
            "pandas",
            "plotly>=4.0.0",
            "scikit-learn>=0.19.0,<0.23.0",
            "scikit-optimize",
            "mlflow",
        ],
        "document": ["sphinx", "sphinx_rtd_theme"],
        "example": [
            "catboost",
            "chainer",
            "lightgbm",
            "mlflow",
            "mpi4py",
            "mxnet",
            "nbval",
            "pytorch-ignite",
            "scikit-image",
            "scikit-learn",
            "thop",
            "torch==1.4.0" if sys.platform == "darwin" else "torch==1.4.0+cpu",
            "torchvision==0.5.0" if sys.platform == "darwin" else "torchvision==0.5.0+cpu",
            "xgboost",
        ]
        + (
            ["allennlp<1", "fastai<2", "pytorch-lightning>=0.7.1"]
            if (3, 5) < sys.version_info[:2] < (3, 8)
            else []
        )
        + (
            ["llvmlite<=0.31.0"] if (3, 5) == sys.version_info[:2] else []
        )  # Newer `llvmlite` is not distributed with wheels for Python 3.5.
        + (
            ["dask[dataframe]", "dask-ml", "keras", "tensorflow>=2.0.0", "tensorflow-datasets"]
            if sys.version_info[:2] < (3, 8)
            else []
        ),
        "experimental": ["redis"],
        "testing": [
            # TODO(toshihikoyanase): Remove the version constraint after resolving the issue
            # https://github.com/optuna/optuna/issues/1000.
            "bokeh<2.0.0",
            "chainer>=5.0.0",
            "cma",
            "fakeredis",
            "fanova",
            "lightgbm",
            "mlflow",
            "mpi4py",
            "mxnet",
            "pandas",
            "plotly>=4.0.0",
            "pytest",
            "pytorch-ignite",
            "scikit-learn>=0.19.0,<0.23.0",
            "scikit-optimize",
            "torch==1.4.0" if sys.platform == "darwin" else "torch==1.4.0+cpu",
            "torchvision==0.5.0" if sys.platform == "darwin" else "torchvision==0.5.0+cpu",
            "xgboost",
        ]
        + (
            ["allennlp<1", "fastai<2", "pytorch-lightning>=0.7.1"]
            if (3, 5) < sys.version_info[:2] < (3, 8)
            else []
        )
        + (
            ["keras", "tensorflow", "tensorflow-datasets"] if sys.version_info[:2] < (3, 8) else []
        ),
    }

    return requirements


def find_any_distribution(pkgs: List[str]) -> Optional[pkg_resources.Distribution]:

    for pkg in pkgs:
        try:
            return pkg_resources.get_distribution(pkg)
        except pkg_resources.DistributionNotFound:
            pass
    return None


pfnopt_pkg = find_any_distribution(["pfnopt"])
if pfnopt_pkg is not None:
    msg = (
        "We detected that PFNOpt is installed in your environment.\n"
        "PFNOpt has been renamed Optuna. Please uninstall the old\n"
        "PFNOpt in advance (e.g. by executing `$ pip uninstall pfnopt`)."
    )
    print(msg)
    exit(1)

setup(
    name="optuna",
    version=get_version(),
    description="A hyperparameter optimization framework",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Takuya Akiba",
    author_email="akiba@preferred.jp",
    url="https://optuna.org/",
    packages=find_packages(),
    package_data={
        "optuna": [
            "storages/rdb/alembic.ini",
            "storages/rdb/alembic/*.*",
            "storages/rdb/alembic/versions/*.*",
        ]
    },
    install_requires=get_install_requires(),
    tests_require=get_tests_require(),
    extras_require=get_extras_require(),
    entry_points={
        "console_scripts": ["optuna = optuna.cli:main"],
        "optuna.command": [
            "create-study = optuna.cli:_CreateStudy",
            "delete-study = optuna.cli:_DeleteStudy",
            "study set-user-attr = optuna.cli:_StudySetUserAttribute",
            "studies = optuna.cli:_Studies",
            "dashboard = optuna.cli:_Dashboard",
            "study optimize = optuna.cli:_StudyOptimize",
            "storage upgrade = optuna.cli:_StorageUpgrade",
        ],
    },
)
