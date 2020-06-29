FROM python:3.7-slim-buster

WORKDIR /usr/src/

RUN pip install --no-cache-dir optuna psycopg2-binary tensorflow keras scikit-learn mlflow

COPY keras_distributed.py .

CMD ["python", "keras_distributed.py"]
