import os
import sys

import pkg_resources
from setuptools import find_packages
from setuptools import setup

from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA


def get_version():
    # type: () -> str

    version_filepath = os.path.join(os.path.dirname(__file__), 'optuna', 'version.py')
    with open(version_filepath) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.strip().split()[-1][1:-1]
    assert False


def get_long_description():
    # type: () -> str

    readme_filepath = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(readme_filepath) as f:
        return f.read()


def get_install_requires():
    # type: () -> List[str]

    return [
        'alembic',
        'cliff',
        'colorlog',
        'numpy',
        'scipy!=1.4.0',
        'sqlalchemy>=1.1.0',
        'tqdm',
        'joblib',
    ]


def get_tests_require():
    # type: () -> List[str]

    return get_extras_require()['testing']


def get_extras_require():
    # type: () -> Dict[str, List[str]]

    requirements = {
        'checking': [
            'autopep8',
            'hacking',
            'mypy',
        ],
        'codecov': [
            'codecov',
            'pytest-cov',
        ],
        'doctest': [
            'pandas',
            'cma',
            'scikit-learn>=0.19.0',
            'plotly>=4.0.0',
        ],
        'document': [
            'sphinx',
            'sphinx_rtd_theme',
        ],
        'example': [
            'catboost',
            'chainer',
            'lightgbm',
            'mlflow',
            'mxnet',
            'pytorch-ignite',
            'scikit-image',
            'scikit-learn',
            'torch',
            'torchvision>=0.5.0',
            'xgboost',
        ] + (['fastai<2'] if (3, 5) < sys.version_info[:2] < (3, 8) else [])
        + ([
            'dask[dataframe]',
            'dask-ml',
            'keras',
            'pytorch-lightning',
            'tensorflow>=2.0.0',
        ] if sys.version_info[:2] < (3, 8) else []),
        'testing': [
            'bokeh',
            'chainer>=5.0.0',
            'cma',
            'lightgbm',
            'mock',
            'mpi4py',
            'mxnet',
            'pandas',
            'plotly>=4.0.0',
            'pytest',
            'pytorch-ignite',
            'scikit-learn>=0.19.0',
            'scikit-optimize',
            'torch',
            'torchvision>=0.5.0',
            'xgboost',
        ] + (['fastai<2'] if (3, 5) < sys.version_info[:2] < (3, 8) else [])
        + ([
            'keras',
            'pytorch-lightning',
            'tensorflow',
            'tensorflow-datasets',
        ] if sys.version_info[:2] < (3, 8) else []),
    }

    return requirements


def find_any_distribution(pkgs):
    # type: (List[str]) -> Optional[pkg_resources.Distribution]

    for pkg in pkgs:
        try:
            return pkg_resources.get_distribution(pkg)
        except pkg_resources.DistributionNotFound:
            pass
    return None


pfnopt_pkg = find_any_distribution(['pfnopt'])
if pfnopt_pkg is not None:
    msg = 'We detected that PFNOpt is installed in your environment.\n' \
        'PFNOpt has been renamed Optuna. Please uninstall the old\n' \
        'PFNOpt in advance (e.g. by executing `$ pip uninstall pfnopt`).'
    print(msg)
    exit(1)

setup(
    name='optuna',
    version=get_version(),
    description='A hyperparameter optimization framework',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Takuya Akiba',
    author_email='akiba@preferred.jp',
    url='https://optuna.org/',
    packages=find_packages(),
    package_data={
        'optuna': [
            'storages/rdb/alembic.ini',
            'storages/rdb/alembic/*.*',
            'storages/rdb/alembic/versions/*.*'
        ]
    },
    install_requires=get_install_requires(),
    tests_require=get_tests_require(),
    extras_require=get_extras_require(),
    entry_points={'console_scripts': ['optuna = optuna.cli:main']})
