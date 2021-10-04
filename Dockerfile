FROM nvcr.io/nvidia/cuda:11.1.1-base-ubi8

MAINTAINER Selbi Nuryyeva <selbi@redhat.com>

# Install Python and pip3
RUN yum -y install python3-devel \
                   python3-pip && \
    yum -y update && \
    yum clean all

# Install PyTorch and other necessary dependencies
RUN pip3 install torch==1.9+cu111 \
                 torchvision==0.10.0+cu111 \
                 torchaudio==0.9.0 \
                 -f https://download.pytorch.org/whl/torch_stable.html
