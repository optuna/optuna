import os
from typing import Dict
from typing import List

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

    # When you update a lower bound of a dependency,
    # please update `test-with-lower` in `.github/workflows/tests.yml` as well.
    requirements = [
        "alembic>=1.5.0",
        "cmaes>=0.9.1",
        "colorlog",
        "numpy",
        "packaging>=20.0",
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
            "distributed",
            "fvcore",
            "lightgbm",
            "matplotlib!=3.6.0",
            "mlflow",
            "pandas",
            "pillow",
            "plotly>=4.9.0",  # optuna/visualization.
            "scikit-learn",
            "scikit-optimize",
            # TODO(not522): Remove the constraint after sphinx_rtd_theme supports Sphinx 6
            "sphinx<6",
            "sphinx-copybutton",
            "sphinx-gallery",
            "sphinx-plotly-directive",
            "sphinx_rtd_theme",
            "torch==1.11.0",
            "torchaudio==0.11.0",
            "torchvision==0.12.0",
        ],
        "integration": [
            "allennlp>=2.2.0; python_version<'3.11'",
            # TODO(c-bata): Remove cached-path after allennllp supports v1.1.3
            "cached-path<=1.1.2; python_version<'3.11'",
            "botorch>=0.4.0,<0.8.0; python_version<'3.11'",
            "catalyst>=21.3; python_version<'3.11'",
            "catboost>=0.26; python_version<'3.11'",
            "chainer>=5.0.0",
            "cma",
            "distributed",
            "fastai; python_version<'3.11'",
            "lightgbm; python_version<'3.11'",
            "mlflow; python_version<'3.11'",
            "mpi4py",
            "mxnet; python_version<'3.11'",
            "pandas",
            "pytorch-ignite; python_version<'3.11'",
            "pytorch-lightning>=1.5.0; python_version<'3.11'",
            "scikit-learn>=0.24.2",
            "scikit-optimize; python_version<'3.11'",
            "shap; python_version<'3.11'",
            "skorch; python_version<'3.11'",
            "tensorflow; python_version<'3.11'",
            "tensorflow-datasets; python_version<'3.11'",
            "torch==1.11.0; python_version<'3.11'",
            "torchaudio==0.11.0; python_version<'3.11'",
            "torchvision==0.12.0; python_version<'3.11'",
            "wandb",
            "xgboost",
        ],
        "optional": [
            "matplotlib!=3.6.0",  # optuna/visualization/matplotlib
            "pandas",  # optuna/study.py
            "plotly>=4.9.0",  # optuna/visualization.
            "redis",  # optuna/storages/redis.py.
            "scikit-learn>=0.24.2",
            # optuna/visualization/param_importances.py.
        ],
        "test": [
            "codecov",
            "fakeredis[lua]",
            "kaleido",
            "pytest",
            "scipy>=1.9.2; python_version>='3.8'",
        ],
    }

    return requirements


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
    packages=find_packages(exclude=("tests", "tests.*", "benchmarks", "benchmarks.*")),
    package_data={
        "optuna": [
            "storages/_rdb/alembic.ini",
            "storages/_rdb/alembic/*.*",
            "storages/_rdb/alembic/versions/*.*",
            "py.typed",
        ]
    },
    python_requires=">=3.7",
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
    entry_points={
        "console_scripts": ["optuna = optuna.cli:main"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
