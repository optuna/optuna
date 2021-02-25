FROM python:3.7-slim-buster

WORKDIR /usr/src/

RUN pip install --no-cache-dir optuna scikit-learn psycopg2-binary

COPY sklearn_distributed.py .
COPY check_study.sh .
