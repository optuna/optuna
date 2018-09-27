# Optuna

## Information for Users

```
$ pip install git+https://github.com/pfnet/optuna.git
```

## Information for Developers

### Format and Lint


We use `flake8` and `autopep8`. To install, run:

```
$ pip install hacking flake8 autopep8
```

To format and make changes to Python codes in place, run the following at the repository root:

```
$ autopep8 . -r --in-place
```

Lint:

```
$ flake8 .
```


### Static Type Checking

We use `mypy`. To install, run:

```
$ pip install mypy
```

To invoke static type checking, run:

```
$ mypy --ignore-missing-imports .
```

We use comment-style type annotation for compatibility with Python 2.

* [PEP484](https://www.python.org/dev/peps/pep-0484/)
* [Syntax cheat sheet](http://mypy.readthedocs.io/en/latest/cheat_sheet.html)
