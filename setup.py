import os
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
        "alembic<=1.4.1",
        "cliff",
        "cmaes>=0.8.2",
        "colorlog",
        "numpy",
        "packaging>=20.0",
        "scipy!=1.4.0",
        "sqlalchemy>=1.1.0,<=1.4.25",
        "tqdm",
        "PyYAML",  # Only used in `optuna/cli.py`.
    ]
    return requirements


def get_tests_require() -> List[str]:

    return get_extras_require()["testing"]


def get_extras_require() -> Dict[str, List[str]]:

    requirements = {
        # TODO(HideakiImamura) Unpin mypy version after fixing "Duplicate modules" error in
        # tutorials.
        "checking": [
            "black==21.9b0",
            # black requires click v7, not v8.
            "click<=7.1.2",
            "hacking",
            "isort",
            "mypy==0.790",
            "blackdoc",
        ],
        "codecov": ["codecov", "pytest-cov"],
        "doctest": [
            "cma",
            "matplotlib>=3.0.0",
            "pandas",
            "plotly>=4.0.0",
            # TODO(nzw0301) remove the constraint of sklearn `1.0.0`
            # after skopt supports sklearn >= 1.0.0
            "scikit-learn>=0.24.2,<1.0.0",
            # TODO(not522): remove this after mlflow is fixed.
            # https://github.com/mlflow/mlflow/pull/5945
            "protobuf<4.0.0",
            "scikit-optimize",
            # See https://github.com/optuna/optuna/issues/3136
            "mlflow<=1.20.2",
        ],
        "document": [
            "sphinx",
            "sphinx_rtd_theme",
            "sphinx-copybutton",
            "sphinx-gallery",
            "sphinx-plotly-directive",
            "pillow",
            "matplotlib",
            # TODO(nzw0301) remove the constraint of sklearn `1.0.0`
            # after skopt supports sklearn >= 1.0.0
            "scikit-learn<1.0.0",
            "plotly>=4.0.0",  # optuna/visualization.
            "pandas",
            "lightgbm<=3.2.1",
            "torch==1.8.0",
            "torchvision==0.9.0",
            "torchaudio==0.8.0",
            "thop",
        ],
        "experimental": ["redis"],
        "testing": [
            # bokeh v1 does not works with MarkupSafe 2.1.
            "MarkupSafe<=2.0.1",
            # bokeh v1 does not works with Jinja v3.
            "Jinja2<3.0.0",
            # TODO(toshihikoyanase): Remove the version constraint after resolving the issue
            # https://github.com/optuna/optuna/issues/1000.
            "bokeh<2.0.0",
            "chainer>=5.0.0",
            "cma",
            "fakeredis<=1.7.1; python_version<'3.8'",
            "fakeredis ; python_version>='3.8'",
            "lightgbm<=3.2.1",
            "matplotlib>=3.0.0",
            # See https://github.com/optuna/optuna/issues/3136
            "mlflow<=1.20.2",
            "mpi4py",
            "mxnet",
            "pandas",
            "plotly>=4.0.0",
            "pytest",
            # TODO(nzw0301) remove the constraint of sklearn `1.0.0`
            # after skopt supports sklearn >= 1.0.0
            "scikit-learn>=0.24.2,<1.0.0",
            # TODO(not522): remove this after mlflow is fixed.
            # https://github.com/mlflow/mlflow/pull/5945
            "protobuf<4.0.0",
            "scikit-optimize",
            "xgboost",
            "tensorflow",
            "tensorflow-datasets",
            "pytorch-ignite",
            "pytorch-lightning>=1.0.2",
            "skorch",
            # The integration module does not work with catalyst v22.
            "catalyst>=21.3,<=21.9",
            "torch==1.8.0 ; sys_platform=='darwin'",
            "torch==1.8.0+cpu ; sys_platform!='darwin'",
            "torchvision==0.9.0 ; sys_platform=='darwin'",
            "torchvision==0.9.0+cpu ; sys_platform!='darwin'",
            "torchaudio==0.8.0",
            # TODO(himkt): Remove `nltk` after solving
            # https://github.com/allenai/allennlp/issues/5521
            "nltk<3.6.6",
            "allennlp>=2.2.0,<2.7.0",
            "botorch>=0.4.0 ; python_version>'3.6'",
            "fastai<=2.5.2",
        ],
        "tests": [
            "fakeredis<=1.7.1; python_version<'3.8'",
            "fakeredis ; python_version>='3.8'",
            "pytest",
        ],
        "optional": [
            # bokeh v1 does not works with MarkupSafe 2.1.
            "MarkupSafe<=2.0.1",
            # bokeh v1 does not works with Jinja v3.
            "Jinja2<3.0.0",
            "bokeh<2.0.0",  # optuna/cli.py, optuna/dashboard.py.
            "matplotlib>=3.0.0",  # optuna/visualization/matplotlib
            "pandas",  # optuna/study.py
            "plotly>=4.0.0",  # optuna/visualization.
            "redis",  # optuna/storages/redis.py.
            # TODO(nzw0301) remove the constraint of sklearn `1.0.0`
            # after skopt supports sklearn >= 1.0.0
            "scikit-learn>=0.24.2,<1.0.0",
            # optuna/visualization/param_importances.py.
        ],
        "integration": [
            # TODO(toshihikoyanase): Remove the version constraint after resolving the issue
            # https://github.com/optuna/optuna/issues/1000.
            "chainer>=5.0.0",
            "cma",
            "lightgbm<=3.2.1",
            # See https://github.com/optuna/optuna/issues/3136
            "mlflow<=1.20.2",
            "wandb",
            "mpi4py",
            "mxnet",
            "pandas",
            # TODO(nzw0301) remove the constraint of sklearn `1.0.0`
            # after skopt supports sklearn >= 1.0.0
            "scikit-learn>=0.24.2,<1.0.0",
            "scikit-optimize",
            "xgboost",
            "tensorflow",
            "tensorflow-datasets",
            "pytorch-ignite",
            "pytorch-lightning>=1.0.2",
            "skorch",
            # The integration module does not work with catalyst v22.
            "catalyst>=21.3,<=21.9",
            "torch==1.8.0 ; sys_platform=='darwin'",
            "torch==1.8.0+cpu ; sys_platform!='darwin'",
            "torchvision==0.9.0 ; sys_platform=='darwin'",
            "torchvision==0.9.0+cpu ; sys_platform!='darwin'",
            "torchaudio==0.8.0",
            # TODO(himkt): Remove `nltk` after solving
            # https://github.com/allenai/allennlp/issues/5521
            "nltk<3.6.6",
            "allennlp>=2.2.0,<2.7.0",
            "botorch>=0.4.0 ; python_version>'3.6'",
            "fastai<=2.5.2",
        ],
        "benchmark": [
            "asv",
            "virtualenv",
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
    tests_require=get_tests_require(),
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
            "dashboard = optuna.cli:_Dashboard",
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
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
