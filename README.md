# Optuna: A hyperparameter optimization framework

*Optuna* is an automatic hyperparameter optimization software framework, particularly designed
for machine learning. It features an imperative, *define-by-run* style user API. Thanks to our
*define-by-run* API, the code written with *Optuna* enjoys high modularity, and the user of
*Optuna* can dynamically construct the search spaces for the hyperparameters. *Optuna* also
features modern functionalities including parallel distributed optimization, premature pruning
of unpromising trials, and web dashboard.

## Installation

To install Optuna, use `pip` as follows:

```
$ pip install git+https://github.com/pfnet/optuna.git
```

Optuna supports Python 2.7 and Python 3.4 or newer.

## Contribution

Any contributions to Optuna are welcome! When you send a pull request, please follow the
[contribution guide](https://github.com/pfnet/optuna/tree/master/CONTRIBUTING.md).

## License

MIT License (see [LICENSE](https://github.com/pfnet/optuna/tree/master/LICENSE)).