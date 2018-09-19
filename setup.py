import os
from setuptools import find_packages
from setuptools import setup
import sys


def get_version():
    version_filepath = os.path.join(os.path.dirname(__file__), 'optuna', 'version.py')
    with open(version_filepath) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.strip().split()[-1][1:-1]
    assert False


def get_install_requires():
    install_requires = [
        'sqlalchemy>=1.1.0', 'numpy', 'scipy', 'six', 'typing', 'cliff', 'colorlog', 'pandas']
    if sys.version_info[0] == 2:
        install_requires.extend(['enum34'])
    return install_requires


def get_tests_require():
    tests_require = ['pytest', 'hacking', 'mock', 'bokeh']
    if sys.version_info[0] == 3:
        # TODO(Yanase): Setting mypy version to 0.620 as a temporal fix
        # for the library's problem in handling NamedTuple since 0.630.
        # The problem is tracked at https://github.com/python/mypy/issues/5640.
        tests_require.append('mypy==0.620')
    return tests_require


setup(
    name='optuna',
    version=get_version(),
    description='',
    author='Takuya Akiba',
    author_email='akiba@preferred.jp',
    packages=find_packages(),
    install_requires=get_install_requires(),
    tests_require=get_tests_require(),
    extras_require={'testing': get_tests_require()},
    entry_points={
        'console_scripts': ['optuna = optuna.cli:main']
    }
)
