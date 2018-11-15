# Optuna: A hyperparameter optimization framework

*Optuna* is an automatic hyperparameter optimization software framework, particularly designed
for machine learning. It features an imperative, *define-by-run* style user API. Thanks to our
*define-by-run* API, the code written with *Optuna* enjoys high modularity, and the user of
*Optuna* can dynamically construct the search spaces for the hyperparameters. *Optuna* also
features modern functionalities including parallel distributed optimization, premature pruning
of unpromising trials, and web dashboard.

## Information for Users

### Installation

To install Optuna, use `pip` as follows:

```
$ pip install git+https://github.com/pfnet/optuna.git
```

Optuna supports Python 2.7 and Python 3.4 or newer.

## Information for Developers

We use `circleci` for continuous integration.
Before creating new PRs, please check your code in your local environment.

First, we use `flake8` for code format checking and `mypy` for static type checking.
To check you code, run:

```
$ circleci build --job checks
```

Note that we use comment-style type annotation for compatibility with Python 2.

* [PEP484](https://www.python.org/dev/peps/pep-0484/)
* [Syntax cheat sheet](http://mypy.readthedocs.io/en/latest/cheat_sheet.html)


Second, we use `pytest` for unit tests. To execute the tests on Python 3.7, run:

```
$ circleci build --job tests-python37
```

Please ensure that your code works without any errors by running following jobs:

- `tests-python27`
- `tests-python34`
- `tests-python35`
- `tests-python36`
- `tests-python37`

Finally, we execute all examples as smoke testing. To execute examples on Python 3.7, run:

```
$ circleci build --job examples-python37
```

Please ensure that your code works without any errors by running following jobs:

- `examples-python27`
- `examples-python34`
- `examples-python35`
- `examples-python36`
- `examples-python37`

## License

MIT License (see `LICENSE` file).