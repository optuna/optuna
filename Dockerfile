ARG PYTHON_VERSION=3.7

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
        if [ "${PYTHON_VERSION}" \< "3.6" ]; then \
            pip install ${PIP_OPTIONS} -e '.[doctest, document, testing]' -f https://download.pytorch.org/whl/torch_stable.html; \
        else \
            pip install ${PIP_OPTIONS} -e '.[checking, doctest, document, testing]' -f https://download.pytorch.org/whl/torch_stable.html; \
        fi \
    else \
        pip install ${PIP_OPTIONS} -e .; \
    fi \
    && pip install ${PIP_OPTIONS} jupyter notebook

# Install RDB bindings.
RUN pip install ${PIP_OPTIONS} PyMySQL cryptography psycopg2-binary

ENV PIP_OPTIONS ""
