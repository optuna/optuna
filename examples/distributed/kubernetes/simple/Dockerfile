FROM python:3.6-slim-buster

WORKDIR /usr/src/

RUN pip install --no-cache-dir optuna scikit-learn psycopg2-binary mlflow

COPY sklearn_distributed.py .

COPY run_mlflow.sh
