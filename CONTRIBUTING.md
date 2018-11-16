# Contributing guidelines

Before creating new PRs, please check your code in your local environment.
We use [CircleCI](https://circleci.com/) for continuous integration, so please setup
[CircleCI Local CLI](https://circleci.com/docs/2.0/local-cli/) in advance.  


## Coding standards

Please apply [flake8](http://flake8.pycqa.org) (code style checker) and
[mypy](http://mypy-lang.org/) (static type checker) as follows:

```
$ circleci build --job checks
```

If any warnings or errors are emitted, please fix them.

Note that we use comment-style type annotation for compatibility with Python 2.

* [PEP484](https://www.python.org/dev/peps/pep-0484/)
* [Syntax cheat sheet](http://mypy.readthedocs.io/en/latest/cheat_sheet.html)


## Tests

We use [pytest](https://docs.pytest.org/) for unit tests. To execute the tests on Python 3.7, run:

```
$ circleci build --job tests-python37
```

Please make sure that following jobs work without any errors in your environment:

- `tests-python27`
- `tests-python34`
- `tests-python35`
- `tests-python36`
- `tests-python37`

We also test Optuna code using examples. The following command executes all examples in
Python 3.7:

```
$ circleci build --job examples-python37
```

Please check your code by running the following jobs:

- `examples-python27`
- `examples-python34`
- `examples-python35`
- `examples-python36`
- `examples-python37`


## Documents

We use [sphinx](http://www.sphinx-doc.org) to generate tutorial and API reference. To make the
documents, run:

```
$ circleci build --job document
```

Please make sure that you can generate the documents without any errors.