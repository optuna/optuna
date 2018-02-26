from setuptools import find_packages
from setuptools import setup


setup(
    name='pfnopt',
    version='0.0.1',
    description='',
    author='Takuya Akiba',
    author_email='akiba@preferred.jp',
    packages=find_packages(),
    install_requires=['numpy', 'six'],
    tests_require=['pytest'],
)
