import os
import sys
from typing import Dict
from typing import List
from typing import Optional

import pkg_resources
from setuptools import find_packages
from setuptools import setup


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
        "cmaes>=0.6.0",
        "colorlog",
        "joblib",
        "numpy",
        "packaging>=20.0",
        "scipy!=1.4.0",
        "sqlalchemy>=1.1.0",
        "tqdm",
    ]


def get_tests_require() -> List[str]:

    return get_extras_require()["testing"]


def get_extras_require() -> Dict[str, List[str]]:

    requirements = {
        "checking": ["black", "hacking", "isort", "mypy==0.782"],
        "codecov": ["codecov", "pytest-cov"],
        "doctest": [
            "cma",
            "matplotlib>=3.0.0",
            "pandas",
            "plotly>=4.0.0",
            "scikit-learn>=0.19.0,<0.23.0",
            "scikit-optimize",
            "mlflow",
        ],
        "document": [
            # TODO(hvy): Unpin `sphinx` version after:
            # https://github.com/sphinx-doc/sphinx/issues/8105.
            "sphinx==3.0.4",
            # As reported in: https://github.com/readthedocs/sphinx_rtd_theme/issues/949,
            # `sphinx_rtd_theme` 0.5.0 is still not compatible with `sphinx` >= 3.0.
            "sphinx_rtd_theme<0.5.0",
            "sphinx-gallery",
            "pillow",
            "matplotlib",
            "scikit-learn",
        ],
        "example": [
            "catboost",
            "chainer",
            "lightgbm",
            "mlflow",
            "mpi4py",
            "mxnet",
            "nbval",
            "scikit-image",
            "scikit-learn>=0.19.0,<0.23.0",  # optuna/visualization/param_importances.py.
            "xgboost",
            "keras",
            "tensorflow>=2.0.0",
            "tensorflow-datasets",
            "pytorch-ignite",
            "pytorch-lightning>=0.8.1",
            "thop",
            "skorch",
            "stable-baselines3>=0.7.0",
            "catalyst",
        ]
        + (
            ["torch==1.6.0", "torchvision==0.7.0"]
            if sys.platform == "darwin"
            else ["torch==1.6.0+cpu", "torchvision==0.7.0+cpu"]
        )
        + (
            [
                "allennlp==1.0.0",
                "fastai<2",
                "dask[dataframe]",
                "dask-ml",
            ]
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
            "lightgbm",
            "matplotlib>=3.0.0",
            "mlflow",
            "mpi4py",
            "mxnet",
            "pandas",
            "plotly>=4.0.0",
            "pytest",
            "scikit-learn>=0.19.0,<0.23.0",
            "scikit-optimize",
            "xgboost",
            "keras",
            "tensorflow",
            "tensorflow-datasets",
            "pytorch-ignite",
            "pytorch-lightning>=0.8.1",
            "skorch",
            "catalyst",
        ]
        + (
            ["torch==1.6.0", "torchvision==0.7.0"]
            if sys.platform == "darwin"
            else ["torch==1.6.0+cpu", "torchvision==0.7.0+cpu"]
        )
        + (["allennlp==1.0.0", "fastai<2"] if sys.version_info[:2] < (3, 8) else []),
        "tests": ["fakeredis", "pytest"],
        "optional": [
            "bokeh<2.0.0",  # optuna/cli.py, optuna/dashboard.py.
            "matplotlib>=3.0.0",  # optuna/visualization/matplotlib
            "pandas",  # optuna/study.py
            "plotly>=4.0.0",  # optuna/visualization.
            "redis",  # optuna/storages/redis.py.
            "scikit-learn>=0.19.0,<0.23.0",  # optuna/visualization/param_importances.py.
        ],
        "integration": [
            # TODO(toshihikoyanase): Remove the version constraint after resolving the issue
            # https://github.com/optuna/optuna/issues/1000.
            "chainer>=5.0.0",
            "cma",
            "lightgbm",
            "mlflow",
            "mpi4py",
            "mxnet",
            "pandas",
            "scikit-learn>=0.19.0,<0.23.0",
            "scikit-optimize",
            "xgboost",
            "keras",
            "tensorflow",
            "tensorflow-datasets",
            "pytorch-ignite",
            "pytorch-lightning>=0.8.1",
            "skorch",
            "catalyst",
        ]
        + (
            ["torch==1.6.0", "torchvision==0.7.0"]
            if sys.platform == "darwin"
            else ["torch==1.6.0+cpu", "torchvision==0.7.0+cpu"]
        )
        + (["allennlp==1.0.0", "fastai<2"] if sys.version_info[:2] < (3, 8) else []),
    }

    return requirements


def find_any_distribution(pkgs: List[str]) -> Optional[pkg_resources.Distribution]:

    for pkg in pkgs:
        try:
            return pkg_resources.get_distribution(pkg)
        except pkg_resources.DistributionNotFound:
            pass
    return None


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
            "storages/_rdb/alembic.ini",
            "storages/_rdb/alembic/*.*",
            "storages/_rdb/alembic/versions/*.*",
            "py.typed",
        ]
    },
    python_requires=">=3.6",
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
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
