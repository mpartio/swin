#FROM nvidia/cuda:12.2.0-devel-rockylinux8
FROM docker.io/pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

#RUN dnf -y install epel-release && dnf config-manager --set-enabled powertools && \
#    dnf -y install libcudnn8 libnccl python39 python39-devel python39-pip vim moreutils

ADD . /swin
WORKDIR /swin

RUN pip install -r requirements.txt

#RUN update-alternatives --set python3 /usr/bin/python3.9 && \
#    python3 -m pip install -r /swin/requirements.txt

