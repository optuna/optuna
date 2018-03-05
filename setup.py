from setuptools import find_packages
from setuptools import setup
import sys


tests_require = ['pytest', 'hacking']
if sys.version_info[0] == 3:
    tests_require.append('mypy')


setup(
    name='pfnopt',
    version='0.0.1',
    description='',
    author='Takuya Akiba',
    author_email='akiba@preferred.jp',
    packages=find_packages(),
    install_requires=['sqlalchemy', 'numpy', 'scipy', 'six', 'typing', 'enum34'],
    tests_require=tests_require,
    extras_require={'testing': tests_require}
)
