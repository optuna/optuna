# Contributing guidelines

Once you send a pull request, it is automatically tested on [CircleCI](https://circleci.com/).
Please setup
[CircleCI Local CLI](https://circleci.com/docs/2.0/local-cli/) and check your code in your local
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


## Testing

The following circleci job runs all unit tests in Python 3.7:

```
$ circleci build --job tests-python37
```

Please make sure that the following jobs work without any errors in your environment:

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

Please make sure that you can generate the documents without any errors.


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