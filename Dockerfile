ARG PYTHON_VERSION=3.8

FROM python:${PYTHON_VERSION}

ENV PIP_OPTIONS "--no-cache-dir --progress-bar off"

RUN apt-get update \
    && apt-get -y install openmpi-bin libopenmpi-dev libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -U pip \
    && pip install ${PIP_OPTIONS} -U setuptools

WORKDIR /workspaces
COPY . .

ARG BUILD_TYPE='dev'

RUN if [ "${BUILD_TYPE}" = "dev" ]; then \
        pip install ${PIP_OPTIONS} -e '.[benchmark, checking, document, integration, optional, test]' --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install ${PIP_OPTIONS} -e .; \
    fi \
    && pip install ${PIP_OPTIONS} jupyter notebook

# Install RDB bindings.
RUN pip install ${PIP_OPTIONS} PyMySQL cryptography psycopg2-binary

ENV PIP_OPTIONS ""
