# Distributed Optimization on Kubernetes

This example is only verified on minikube.

This example's code is based on ../../pytorch_lightning_simple.py example with the following changes:

1. It gives a name to the study and sets `load_if_exists` to `True` in order to avoid errors when the code is run from multiple workers.
2. It sets the storage address to the postgres pod deployed with the workers.
3. It uses `MLfloatCallback`.

In order to run this example you have to do the following steps:

1. (Optional) If run locally inside [minikube](https://github.com/kubernetes/minikube) you have to use the Docker daemon inside of it:

```bash
$ eval $(minikube docker-env)
```

2. Build and tag the example docker image:

```bash
$ docker build -t optuna-kubernetes-mlflow:example .
```

3. Apply the kubernetes manifests:

```bash
$ kubectl apply -f k8s-manifest.yaml
```

4. Track the study by checking MLflow dashboard:

You can tell the IP address of MLflow dashboard as follows:

```bash
$ minikube service mlflow --url
```

Also, if you want to track the progress of each worker by checking their logs directly:

```bash
$ kubectl logs worker-<pod id>
```
