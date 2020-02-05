ARG PYTHON_VERSION=3.7
ARG BUILD_TYPE=''

FROM python:${PYTHON_VERSION}

RUN apt-get update \
    && apt-get -y install openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -U 'pip<20' \
    && pip install --no-cache-dir --progress-bar off -U setuptools

WORKDIR /workspaces
COPY . .

RUN if [ "${BUILD_TYPE}" = "dev" ]; then \
        pip install --no-cache-dir -e '.[checking, doctest, document, example, testing]'; pip install jupyter notebook; \
    else \
        pip install --no-cache-dir -e .; pip install jupyter notebook; \
    fi
