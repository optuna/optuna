import os
from setuptools import find_packages
from setuptools import setup
import sys


def get_version():
    version_filepath = os.path.join(os.path.dirname(__file__), 'pfnopt', 'version.py')
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
        tests_require.append('mypy')
    return tests_require


setup(
    name='pfnopt',
    version=get_version(),
    description='',
    author='Takuya Akiba',
    author_email='akiba@preferred.jp',
    packages=find_packages(),
    install_requires=get_install_requires(),
    tests_require=get_tests_require(),
    extras_require={'testing': get_tests_require()},
    entry_points={
        'console_scripts': ['pfnopt = pfnopt.cli:main']
    }
)
