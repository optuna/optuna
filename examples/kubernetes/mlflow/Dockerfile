# NOTE(crcrpar): optuna/optuna:py3.7-dev can be used as base instead.
FROM python:3.7-slim-buster

WORKDIR /usr/src/

RUN pip install --no-cache-dir optuna psycopg2-binary mlflow \
    && pip install --no-cache-dir \
        torch==1.5.1+cpu \
        torchvision==0.6.1+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install pytorch-lightning

COPY pytorch_lightning_distributed.py .
COPY check_study.sh .
