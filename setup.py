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

    requirements = [
        "alembic>=1.5.0",
        "cliff",
        "cmaes>=0.8.2",
        "colorlog",
        # TODO(HideakiImamura): remove this after the fix by `cliff` or `stevedore`
        "importlib-metadata<5.0.0",
        "numpy",
        "packaging>=20.0",
        # TODO(kstoneriv3): remove this after deprecation of Python 3.6
        "scipy!=1.4.0,<1.9.0" if sys.version[:3] == "3.6" else "scipy>=1.7.0,<1.9.0",
        "sqlalchemy>=1.3.0",
        "tqdm",
        "PyYAML",  # Only used in `optuna/cli.py`.
    ]
    return requirements


def get_extras_require() -> Dict[str, List[str]]:

    requirements = {
        "benchmark": [
            "asv>=0.5.0",
            "botorch",
            "cma",
            "scikit-optimize",
            "virtualenv",
        ],
        "checking": [
            "black",
            "blackdoc",
            "hacking",
            "isort",
            "mypy",
            "types-PyYAML",
            "types-redis",
            "types-setuptools",
            "typing_extensions>=3.10.0.0",
        ],
        "document": [
            "cma",
            "lightgbm",
            "matplotlib!=3.6.0",
            "mlflow",
            # TODO(nzw0301): Remove onnx if thop adds onnx to its dependencies.
            "onnx",
            "pandas",
            "pillow",
            "plotly>=4.0.0",  # optuna/visualization.
            # TODO(nzw0301): Remove protobuf after
            # https://github.com/onnx/onnx/issues/4239 is resolved.
            "protobuf<=3.20.1",
            "scikit-learn",
            "scikit-optimize",
            "sphinx",
            "sphinx-copybutton",
            "sphinx-gallery",
            "sphinx-plotly-directive",
            "sphinx_rtd_theme",
            "thop",
            "torch==1.11.0 ; python_version>'3.6'",
            "torchaudio==0.11.0 ; python_version>'3.6'",
            "torchvision==0.12.0 ; python_version>'3.6'",
        ],
        "integration": [
            "allennlp>=2.2.0 ; python_version>'3.6'",
            # TODO(c-bata): Remove cached-path after allennllp supports v1.1.3
            "cached-path<=1.1.2 ; python_version>'3.6'",
            "botorch>=0.4.0 ; python_version>'3.6'",
            "catalyst>=21.3 ; python_version>'3.6'",
            "catboost>=0.26",
            "chainer>=5.0.0",
            "cma",
            "fastai ; python_version>'3.6'",
            "lightgbm",
            "mlflow",
            "mpi4py",
            "mxnet",
            "pandas",
            "pytorch-ignite ; python_version>'3.6'",
            "pytorch-lightning>=1.5.0 ; python_version>'3.6'",
            "scikit-learn>=0.24.2",
            "scikit-optimize",
            "shap",
            "skorch ; python_version>'3.6'",
            "tensorflow ; python_version>'3.6'",
            "tensorflow-datasets",
            "torch==1.11.0 ; python_version>'3.6'",
            "torchaudio==0.11.0 ; python_version>'3.6'",
            "torchvision==0.12.0 ; python_version>'3.6'",
            "wandb",
            "xgboost",
        ],
        "optional": [
            "matplotlib!=3.6.0",  # optuna/visualization/matplotlib
            "pandas",  # optuna/study.py
            "plotly>=4.0.0",  # optuna/visualization.
            "redis",  # optuna/storages/redis.py.
            "scikit-learn>=0.24.2",
            # optuna/visualization/param_importances.py.
        ],
        "test": [
            "codecov",
            "fakeredis<=1.7.1; python_version<'3.7'",
            "fakeredis ; python_version>='3.7'",
            "kaleido",
            "pytest",
        ],
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
    project_urls={
        "Source": "https://github.com/optuna/optuna",
        "Documentation": "https://optuna.readthedocs.io",
        "Bug Tracker": "https://github.com/optuna/optuna/issues",
    },
    packages=find_packages(exclude=("tests", "tests.*", "benchmarks")),
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
    extras_require=get_extras_require(),
    entry_points={
        "console_scripts": ["optuna = optuna.cli:main"],
        "optuna.command": [
            "create-study = optuna.cli:_CreateStudy",
            "delete-study = optuna.cli:_DeleteStudy",
            "study set-user-attr = optuna.cli:_StudySetUserAttribute",
            "studies = optuna.cli:_Studies",
            "trials = optuna.cli:_Trials",
            "best-trial = optuna.cli:_BestTrial",
            "best-trials = optuna.cli:_BestTrials",
            "study optimize = optuna.cli:_StudyOptimize",
            "storage upgrade = optuna.cli:_StorageUpgrade",
            "ask = optuna.cli:_Ask",
            "tell = optuna.cli:_Tell",
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
