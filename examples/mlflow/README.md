# Tracking optimization process with MLflow

![mlflow-ui](https://user-images.githubusercontent.com/17039389/70850501-4cdefd80-1ece-11ea-9018-e47363c81f08.gif)

Optuna example that optimizes a neural network regressor for the
wine quality dataset using Keras and records hyperparameters and metrics using MLflow.

In this example, we optimize the learning rate and momentum of
a stochastic gradient descent optimizer to minimize the validation mean squared error
for the wine quality regression.

You can run this example as follows:

```
$ python keras_mlflow.py
```

After the script finishes, run the MLflow UI:

```
$ mlflow ui
```

and view the optimization results at http://127.0.0.1:5000.
