From python:3.5-slim-buster

WORKDIR /usr/src/

RUN pip install --no-cache-dir optuna scikit-learn psycopg2-binary

COPY sklearn_distributed.py .

CMD ["python", "sklearn_distributed.py"]

