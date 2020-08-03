#!/bin/bash

function run {
  IsMinikube=$1
  IMAGE_NAME=$2

  if [ "${IsMinikube}" = True ]; then
    eval "$(minikube docker-env)"
  fi

  # Build Docker Image
  echo "Building Docker Image ${IMAGE_NAME} \n"
  docker image build -t "${IMAGE_NAME}" .

  # Push docker image to proper container registry if using cloud provider
  if [ "${IsMinikube}" = False ]; then
    docker push "${IMAGE_NAME}"
  fi

  # Deploy to cluster
  echo "\nDeploying example to cluster \n"
  kubectl apply -f k8s-manifests.yaml
}

IsMinikube=$1
IMAGE_NAME=$2
run ${IsMinikube} ${IMAGE_NAME}
