ARG PYTHON_VERSION=3.7

FROM python:${PYTHON_VERSION}

RUN apt-get update \
    && apt-get -y install openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -U 'pip<20' \
    && pip install --no-cache-dir --progress-bar off -U setuptools

WORKDIR /workspaces
COPY . .

RUN pip install --no-cache-dir -e '.[checking,doctest,document,testing]' \
    && pip install jupyter notebook
