FROM nvidia/cuda:7.5-cudnn5-devel

MAINTAINER Nikki Aldeborgh <nikki.aldeborgh@digitalglobe.com>

RUN apt-get -y update && apt-get -y install \ 
    python \
    python-pip \
    python-six \
    python-dev \
    python-setuptools \
    libatlas-base-dev \ 
    gfortran \
    libyaml-dev \
    gdal-bin \
    python-gdal \
    libhdf5-serial-dev \
    vim \
    ipython \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install awscli Theano==0.9.0 Keras==1.2.2 geojson h5py Pillow==2.6.0 

ARG PROTOUSER 
ARG PROTOPASSWORD
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_SESSION_TOKEN

RUN git clone https://${PROTOUSER}:${PROTOPASSWORD}@github.com/digitalglobe/protogen && \
    cd protogen && \
    git checkout dev && \
    python setup.py install && \
    cd ..

RUN aws s3 cp s3://gbd-customer-data/58600248-2927-4523-b44b-5fec3d278c09/platform-stories/ship-detector/model/model.h5 /

COPY ./bin /
COPY .theanorc /root/.theanorc
COPY keras.json /root/.keras/keras.json
