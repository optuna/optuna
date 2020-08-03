#!/bin/sh
optuna studies --storage "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}" | grep k8s_mlflow > /dev/null
echo $?
