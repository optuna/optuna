FROM python:3.5-slim-buster


WORKDIR /usr/src/

RUN pip install --no-cache-dir optuna scikit-learn psycopg2-binary joblibspark


COPY sklearn_distributed_spark.py .
COPY check_study.sh .