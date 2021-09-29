FROM condaforge/mambaforge:4.10.3-5

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV TZ=Europe/Berlin
ENV TERM xterm-256color
ARG PYTHON_VERSION=3.6
ENV PYTHON_VERSION=$PYTHON_VERSION
ARG N_JOBS=8
ENV N_JOBS=$N_JOBS
ENV MAKEFLAGS=-j${N_JOBS}

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
    && apt-get install -y --no-install-recommends --allow-unauthenticated apt-utils \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
    zip \
    gzip \
    make \
    automake \
    gcc \
    build-essential \
    g++ \
    cpp \
    libc6-dev \
    man-db \
    autoconf \
    pkg-config \
    unzip \
    libffi-dev \
    software-properties-common \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /fever
WORKDIR /fever

COPY make_replacement.sh /usr/local/bin/make
RUN bash -c 'chmod +x /usr/local/bin/make'

COPY environment.yml /fever/

RUN mamba env create -f environment.yml

SHELL ["bash", "-lc"]
ENV PATH /opt/conda/envs/fever/bin:$PATH

RUN /bin/bash -c "source activate fever"

RUN python -c "import nltk; nltk.download('punkt')"

RUN mkdir -pv src configs

ADD src src
ADD configs configs

ADD predict.sh .

ENV PYTHONPATH src
ENV FLASK_APP sample_application:my_sample_fever

#ENTRYPOINT ["/bin/bash","-c"]
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "--call", "sample_application:my_sample_fever"]
