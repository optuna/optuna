# Distributed Optimization on Kubernetes

This example's code is mostly the same as the sklearn_simple.py example, 
except for two things:

1 - It gives a name to the study and sets load_if_exists to True
in order to avoid errors when the code is run from multiple workers.

2 - It sets the storage address to the postgres pod deployed with the workers.

In order to run this example you have to do the following steps:

Run `run.sh` which takes two arguments `$IsMinikube` and `$IMAGE_NAME`

- If you want to run locally in minikube run the following command

 ```bash
$ bash run.sh True optuna-kubernetes:example
 ```

- If you want to run in cloud, please change the IMAGE_NAME accordingly in k8s-manifest.yaml and run as follows. Also please make sure that you kubernetes context is set correctly.

 ```bash
$ bash run.sh False $IMAGE_NAME
 ```

- Track the progress of each worker by checking their logs:

```bash
kubectl logs worker-<pod id>
```
