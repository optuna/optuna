# Distributed Optimization on Kubernetes

This folder contains two kinds of examples with Kubernetes: one is based on `sklearn_simple.py` and the other is based on `pytorch_lightning_simple.py` with MLflow.

Currently, both `./simple/sklearn_distributed.py` and `./mlflow/pytorch_lightning_distributed.py` use POSTGRESQL for their backend of `optuna.Study.optimize` to be parallelized.  
Though we do not use it for MLflow records.  Of course, you can use POSTGRESQL as backend store of MLflow (https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded), current example uses HTTP server.
