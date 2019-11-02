# Distributed Optimization on Kubernetes

This example's code is mostly the same as the sklearn_simple.py example, 
except for two things:

1 - It gives a name to the study and sets load_if_exists to True
in order to avoid errors when the code is run from multiple workers.

2 - It sets the storage address to the postgres pod deployed with the workers.

In order to run this example you have to do the following steps:

1 - (Optional) If run locally inside [minikube](https://github.com/kubernetes/minikube) 
you have to use the Docker daemon inside of it: 

```bash
eval $(minikube docker-env)
```

2 - Build and tag the example docker image:

```bash
docker image build -t optuna-kubernetes:example .
```

3 - Apply the kubernetes manifests:

```bash
kubectl apply -f k8s-manifests.yaml
```

4 - Track the progress of each worker by checking their logs:

```bash
kubectl logs worker-<pod id>
```


