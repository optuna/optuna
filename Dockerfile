ARG PYTHON_VERSION=3.7

FROM python:${PYTHON_VERSION}

ENV PIP_OPTIONS "--no-cache-dir --progress-bar off"

RUN apt-get update \
    && apt-get -y install openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && wget -q http://prdownloads.sourceforge.net/swig/swig-3.0.12.tar.gz \
    && tar -zxvf swig-3.0.12.tar.gz \
    && cd swig-3.0.12 \
    && ./configure \
    && make --silent -j \
    && make install --silent \
    && cd .. \
    && rm -rf swig-3.0.12 \
    && rm swig-3.0.12.tar.gz \
    && swig -version \
    && pip install --no-cache-dir -U pip \
    && pip install ${PIP_OPTIONS} -U setuptools \
    && pip install ${PIP_OPTIONS} Cython  # for automl/ConfigSpace

WORKDIR /workspaces
COPY . .

ARG BUILD_TYPE='dev'

RUN if [ "${BUILD_TYPE}" = "dev" ]; then \
        if [ "${PYTHON_VERSION}" \< "3.6" ]; then \
            pip install ${PIP_OPTIONS} -e '.[doctest, document, example, testing]' -f https://download.pytorch.org/whl/torch_stable.html; \
        else \
            pip install ${PIP_OPTIONS} -e '.[checking, doctest, document, example, testing]' -f https://download.pytorch.org/whl/torch_stable.html; \
        fi \
    else \
        pip install ${PIP_OPTIONS} -e .; \
    fi \
    && pip install ${PIP_OPTIONS} jupyter notebook

ENV PIP_OPTIONS ""
