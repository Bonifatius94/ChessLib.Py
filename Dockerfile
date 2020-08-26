FROM ubuntu:18.04

# configure build settings
ARG GITHUB_REPO=ChessAI.Py
ARG SRC_DIR=/home/src/$GITHUB_REPO
ARG BUILD_DIR=/home/build/

# install build dependencies
RUN apt-get update && \
    apt-get install -y git build-essential cmake python3 python3-dev python3-pip && \
    pip3 install numpy && \
    rm -rf /var/lib/apt/lists/*

# download and build source code
RUN git clone https://github.com/Bonifatius94/$GITHUB_REPO /home/src/ && \
    cd $BUILD_DIR && cmake ../src/$GITHUB_REPO && \
    make -j4 && \
    rm -rf $SRC_DIR

# TODO: add deployment commands
