# Tracking optimization process with MLflow

Optuna example that optimizes a neural network regressor for the
wine quality dataset using Keras and records hyperparamters and metrics using MLflow.

In this example, we optimize the `learning rate` and `momentum` of
stochastic gradient descent optimizer to minimize the validation mean squared error
for the wine quality regression.

We have the following two ways to execute this example:

(1) Excute code directly.

```
$ python keras_mlflow.py
```

(2) Execute through CLI.

```
$ STUDY_NAME=`optuna create-study --direction minimize --storage sqlite:///example.db`
$ optuna study optimize keras_mlflow.py objective --n-trials=100 \
         --study $STUDY_NAME --storage sqlite:///example.db
```

After the script finishes, run the MLflow UI:

```
$ mlflow ui
```

and view the optimization results at http://127.0.0.1:5000.

![mlflow-ui](https://user-images.githubusercontent.com/17039389/70850501-4cdefd80-1ece-11ea-9018-e47363c81f08.gif)
