FROM docker.io/pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

RUN apt -y update && apt -y install git vim && apt clean all

RUN cd / && git clone --verbose https://github.com/mpartio/swin.git

WORKDIR /swin

RUN pip install -r requirements.txt
