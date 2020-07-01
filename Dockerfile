ARG PYTHON_VERSION=3.7

FROM python:${PYTHON_VERSION}

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
    && pip install --no-cache-dir --progress-bar off -U setuptools \
    && pip install --no-cache-dir --progress-bar off Cython  # for automl/ConfigSpace

WORKDIR /workspaces
COPY . .

ARG BUILD_TYPE='dev'

RUN if [ "${BUILD_TYPE}" = "dev" ]; then \
        if [ "${PYTHON_VERSION}" \< "3.6" ]; then \
            pip install --no-cache-dir --progress-bar off -e '.[doctest, document, example, testing]' -f https://download.pytorch.org/whl/torch_stable.html; \
        else \
            pip install --no-cache-dir --progress-bar off -e '.[checking, doctest, document, example, testing]' -f https://download.pytorch.org/whl/torch_stable.html; \
        fi \
    else \
        pip install --no-cache-dir --progress-bar off -e .; \
    fi \
    && pip install --no-cache-dir --progress-bar off jupyter notebook
