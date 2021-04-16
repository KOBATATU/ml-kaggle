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
    python3-setuptools

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*



# mecab-ipadic-neologd install
WORKDIR /opt
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
WORKDIR /opt/mecab-ipadic-neologd
RUN ./bin/install-mecab-ipadic-neologd -n -y
WORKDIR /opt
RUN rm -rf mecab-ipadic-neologd


# Set up Jupyter Notebook config
ENV CONFIG /root/.jupyter/jupyter_notebook_config.py
ENV CONFIG_IPYTHON /root/.ipython/profile_default/ipython_config.py

# install additional packages
ADD requirements.txt requirements.txt
RUN pip install -U pip && pip install -r requirements.txt



RUN jupyter notebook --generate-config --allow-root && \
    ipython profile create

RUN echo "c.NotebookApp.ip = '0.0.0.0'" >>${CONFIG} && \
    echo "c.NotebookApp.port = 8888" >>${CONFIG} && \
    echo "c.NotebookApp.open_browser = False" >>${CONFIG} && \
    echo "c.NotebookApp.iopub_data_rate_limit=10000000000" >>${CONFIG} && \
    echo "c.MultiKernelManager.default_kernel_name = 'python3'" >>${CONFIG}

RUN echo "c.InteractiveShellApp.exec_lines = ['%matplotlib inline']" >>${CONFIG_IPYTHON}



RUN apt-get install -y unzip

ENV NOTO_DIR /usr/share/fonts/opentype/notosans
RUN mkdir -p ${NOTO_DIR} &&\
    wget -q https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip -O noto.zip &&\
    unzip ./noto.zip -d ${NOTO_DIR}/ &&\
    chmod a+r ${NOTO_DIR}/NotoSans* &&\
    rm ./noto.zip

ADD matplotlibrc /usr/local/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc


# Port
EXPOSE 8888

# Mount
VOLUME /notebooks

# Run Jupyter Notebook
WORKDIR "/notebooks"
CMD ["jupyter","notebook", "--allow-root"]