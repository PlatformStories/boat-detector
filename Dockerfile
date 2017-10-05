FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04

MAINTAINER Nikki Aldeborgh <nikki.aldeborgh@digitalglobe.com>

RUN apt-get -y update && apt-get -y install \
    python \
    build-essential \
    python-pip \
    python-dev \
    ipython \
    python-scipy \
    python-numpy \
    libopencv-dev \
    python-opencv \
    gdal-bin \
    python-gdal \
    vim \
    wget \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install awscli Keras==2.0.6 geojson h5py tensorflow-gpu==1.2.1 utm

ARG PROTOUSER
ARG PROTOPASSWORD
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_SESSION_TOKEN

RUN git clone https://${PROTOUSER}:${PROTOPASSWORD}@github.com/digitalglobe/protogen && \
    cd protogen && \
    python setup.py install && \
    cd ..

RUN aws s3 cp s3://gbd-customer-data/32cbab7a-4307-40c8-bb31-e2de32f940c2/platform-stories/boat-detector/models/8-21-2017/model-epochs-20-lr-0.0001-8-21-2017.h5 /model.h5
RUN aws s3 cp --recursive s3://gbd-customer-data/32cbab7a-4307-40c8-bb31-e2de32f940c2/platform-stories/boat-detector/water-polygons /water-polygons

COPY ./bin /
COPY keras.json /root/.keras/keras.json

