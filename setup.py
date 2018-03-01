from setuptools import find_packages
from setuptools import setup


tests_require = ['pytest', 'hacking', 'mypy']


setup(
    name='pfnopt',
    version='0.0.1',
    description='',
    author='Takuya Akiba',
    author_email='akiba@preferred.jp',
    packages=find_packages(),
    install_requires=['sqlalchemy', 'numpy', 'scipy', 'six'],
    tests_require=tests_require,
    extras_require={'testing': tests_require}
)
