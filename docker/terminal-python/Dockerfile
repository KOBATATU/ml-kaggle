# Python3.6
FROM python:3.6

# Install dependencies
RUN apt-get update && apt-get install -y \
    libblas-dev \
    liblapack-dev\
    libatlas-base-dev \
    mecab \
    mecab-naist-jdic \
    mecab-ipadic-utf8 \
    swig \
    libmecab-dev \
    gfortran \
    sudo \
    cmake \
    python3-setuptools \
    wget

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python library for Data Science
RUN pip --no-cache-dir install mecab-python3 

# mecab-ipadic-neologd install
WORKDIR /opt
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
WORKDIR /opt/mecab-ipadic-neologd
RUN ./bin/install-mecab-ipadic-neologd -n -y
WORKDIR /opt
RUN rm -rf mecab-ipadic-neologd

# ADD matplotlib turn to japanese
ADD matplotlibrc /usr/local/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc

#ADD python packages
ADD requirements.txt requirements.txt
RUN pip install -U pip && pip install -r requirements.txt

ENV NOTO_DIR /usr/share/fonts/opentype/notosans
RUN mkdir -p ${NOTO_DIR} &&\
    wget -q https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip -O noto.zip &&\
    unzip ./noto.zip -d ${NOTO_DIR}/ &&\
    chmod a+r ${NOTO_DIR}/NotoSans* &&\
    rm ./noto.zip

ADD matplotlibrc /usr/local/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc



WORKDIR /code

RUN pip install \
    xonsh \
    prompt_toolkit \
    xontrib-fzf-widgets



CMD ["xonsh"]