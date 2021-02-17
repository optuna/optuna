# Distributed Optimization on Kubernetes

This example is only verified on minikube.

This example's code is based on ../../pytorch/pytorch_lightning_simple.py example with the following changes:

1. It gives a name to the study and sets `load_if_exists` to `True` in order to avoid errors when the code is run from multiple workers.
2. It sets the storage address to the postgres pod deployed with the workers.
3. It uses `MLfloatCallback`.

In order to run this example you have to do the following steps:

First run `run.sh` which takes two arguments `$IsMinikube` and `$IMAGE_NAME`

- If you want to run locally in minikube run the following command

 ```bash
$ sh run.sh True optuna-kubernetes-mlflow:example
 ```

- If you want to run in cloud, please change the `IMAGE_NAME` accordingly in k8s-manifest.yaml and run as follows. Also please make sure that your kubernetes context is set correctly.

 ```bash
$ sh run.sh False $IMAGE_NAME
 ```

- Track the study by checking MLflow dashboard. You can tell the IP address of MLflow dashboard as follows:

```bash
$ minikube service mlflow --url
```

- Also, if you want to track the progress of each worker by checking their logs directly:

```bash
$ kubectl logs worker-<pod id>
```
