# Contributing guidelines

Once you send a pull request, it is automatically tested on [CircleCI](https://circleci.com/).
By setting up the
[CircleCI Local CLI](https://circleci.com/docs/2.0/local-cli/), you can check your code in your local
environment before submitting the pull request.


## Coding standards

The following circleci job runs code checking:

```
$ circleci build --job checks
```

The above job contains following checkers:
- [flake8](http://flake8.pycqa.org)
- [autopep8](https://github.com/hhatto/autopep8)
- [mypy](http://mypy-lang.org/)

If any warnings or errors are emitted, please fix them.

Note that we use comment-style type annotation for compatibility with Python 2.

* [PEP484](https://www.python.org/dev/peps/pep-0484/)
* [Syntax cheat sheet](http://mypy.readthedocs.io/en/latest/cheat_sheet.html)

Please see also our [Coding Style Conventions](https://github.com/optuna/optuna/wiki/Coding-Style-Conventions).

## Testing

When adding a new feature or fixing a bug, you also need to write sufficient test code.
We use [pytest](https://pytest.org/) as the testing framework and
unit tests are stored under the [tests directory](./tests).

You can run your tests as follows:
```console
// Run all the unit tests.
$ pytest

// Run all the unit tests defined in the specified test file.
$ pytest tests/${TARGET_TEST_FILE_NAME}
```


### CircleCI Local CLI

The following circleci job runs all unit tests in Python 3.7:

```console
// Note that this job will download several hundred megabytes of data to
// install all the packages required for testing,
// and take several tens of minutes to complete all tests.
$ circleci build --job tests-python37
```

You can run tests and examples for each Python version using the following jobs:

- `tests-python27`
- `tests-python35`
- `tests-python36`
- `tests-python37`
- `examples-python27`
- `examples-python35`
- `examples-python36`
- `examples-python37`


In addition, to check the documents, run:

```
$ circleci build --job document
```


## Documentation

When adding a new feature to the framework, you also need to document it in the reference.
The documentation source is stored under [docs directory](./docs) and written in
[reStructuredText format](http://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html).

To build the documentation, you need to install [Sphinx](http://www.sphinx-doc.org):

```
$ pip install sphinx sphinx_rtd_theme
```

Then you can build the documentation in HTML format locally:

```
$ cd docs
$ make html
```

HTML files are generated under `build/html` directory. Open `index.html` with the browser and see
if it is rendered as expected.

Note that docstrings (documentation comments in the source code) are collected from the installed
Optuna module. If you modified docstrings, make sure to install the module (e.g.,
using `pip install -e .`) before building the documentation.
