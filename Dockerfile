# get the current pytorch image
FROM pytorch/pytorch

ARG USER_ID
ARG GROUP_ID

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# set the working directory and copy everything to the docker file
WORKDIR ./
COPY ./requirements.txt ./

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN apt-get update && apt-get -y upgrade && apt-get -y install git nano build-essential cmake libboost-all-dev libgtk-3-dev git-lfs
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

USER user
