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


tests_require = ['pytest', 'hacking', 'mock', 'bokeh']
if sys.version_info[0] == 3:
    tests_require.append('mypy')


setup(
    name='pfnopt',
    version=get_version(),
    description='',
    author='Takuya Akiba',
    author_email='akiba@preferred.jp',
    packages=find_packages(),
    install_requires=[
        'sqlalchemy>=1.1.0', 'numpy', 'scipy', 'six', 'typing', 'enum34', 'cliff', 'colorlog',
        'importlib'],
    tests_require=tests_require,
    extras_require={'testing': tests_require},
    entry_points={
        'console_scripts': ['pfnopt = pfnopt.cli:main']
    }
)
