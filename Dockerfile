# ================================================ #
#     Build Engine for Python Extension C-Libs     #
# ================================================ #

# use the Ubuntu 18.04 LTS image as base
FROM ubuntu:18.04

# install build dependencies: c/c++ pyhton3 numpy
RUN apt-get update && \
    apt-get install -y build-essential python3 python3-dev python3-pip && \
    pip3 install numpy && \
    rm -rf /var/lib/apt/lists/*

# configure build settings
ARG SRC_FOLDER=ChessLib
ARG SRC_ROOT=/home/src/$SRC_FOLDER

# copy the source code
ADD ./$SRC_FOLDER $SRC_ROOT

# move inside sources folder
WORKDIR $SRC_ROOT

# build and install the Python extension
# user-only installation, no system installation -> fixes privileges issues
RUN python3 setup.py install --user

# run unit test of Python extension
RUN python3 test.py

# TODO: add deployment commands (packaging binaries, uploading to artifactory, ...)

# ================================================ #
#            Marco Tröster, 2020-09-04             #
# ================================================ #
