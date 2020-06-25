ARG PYTHON_VERSION=3.7

FROM python:${PYTHON_VERSION}

RUN apt-get update \
    && apt-get -y install openmpi-bin libopenmpi-dev \
    && apt-get -y install swig  \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir --progress-bar off -U setuptools

WORKDIR /workspaces
COPY . .

ARG BUILD_TYPE='dev'

RUN if [ "${BUILD_TYPE}" = "dev" ]; then \
        if [ "${PYTHON_VERSION}" \< "3.6" ]; then \
            pip install --no-cache-dir -e '.[doctest, document, example, testing]' -f https://download.pytorch.org/whl/torch_stable.html; \
        else \
            pip install --no-cache-dir -e '.[checking, doctest, document, example, testing]' -f https://download.pytorch.org/whl/torch_stable.html; \
        fi \
    else \
        pip install --no-cache-dir -e .; \
    fi \
    && pip install --no-cache-dir jupyter notebook
