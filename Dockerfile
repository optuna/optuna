ARG PYTHON_VERSION=3.7
ARG BASE_IMAGE=python

FROM ${BASE_IMAGE}:${PYTHON_VERSION}

RUN apt-get update \
    && apt-get -y install openmpi-bin libopenmpi-dev \
    && apt-get -y install swig  \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir --progress-bar off -U setuptools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces
COPY . .

ARG BUILD_TYPE='dev'

RUN if [ "${BUILD_TYPE}" = "dev" ]; then \
        apt-get update \
        && apt-get -y install default-mysql-server default-mysql-client \
        && apt-get -y install postgresql postgresql-client \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; \
    fi

RUN if [ "${BUILD_TYPE}" = "dev" ]; then \
        if [ "${PYTHON_VERSION}" \< "3.6" ]; then \
            pip install --no-cache-dir -e '.[doctest, document, example, testing]' PyMySQL cryptography psycopg2-binary -f https://download.pytorch.org/whl/torch_stable.html; \
        else \
            pip install --no-cache-dir -e '.[checking, doctest, document, example, testing]' PyMySQL cryptography psycopg2-binary -f https://download.pytorch.org/whl/torch_stable.html; \
        fi \
    else \
        pip install --no-cache-dir -e .; \
    fi \
    && pip install --no-cache-dir jupyter notebook
