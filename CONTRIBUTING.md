# Contributing guidelines

Before creating new PRs, please check your code in your local environment.
We use [CircleCI](https://circleci.com/) for continuous integration, so please setup
[CircleCI Local CLI](https://circleci.com/docs/2.0/local-cli/) in advance.  

## Coding standards

We use `flake8` for code format checking and `mypy` for static type checking.
To check you code, run:

```
$ circleci build --job checks
```

Note that we use comment-style type annotation for compatibility with Python 2.

* [PEP484](https://www.python.org/dev/peps/pep-0484/)
* [Syntax cheat sheet](http://mypy.readthedocs.io/en/latest/cheat_sheet.html)


## Tests

We use `pytest` for unit tests. To execute the tests on Python 3.7, run:

```
$ circleci build --job tests-python37
```

Please ensure that your code works without any errors by running following jobs:

- `tests-python27`
- `tests-python34`
- `tests-python35`
- `tests-python36`
- `tests-python37`

We execute all examples as smoke testing. To execute examples on Python 3.7, run:

```
$ circleci build --job examples-python37
```

Please ensure that your code works without any errors by running following jobs:

- `examples-python27`
- `examples-python34`
- `examples-python35`
- `examples-python36`
- `examples-python37`
