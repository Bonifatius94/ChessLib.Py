
# ================================================================================= #
# MIT License                                                                       #
#                                                                                   #
# Copyright(c) 2020 Marco Tröster                                                   #
#                                                                                   #
# Permission is hereby granted, free of charge, to any person obtaining a copy      #
# of this software and associated documentation files (the "Software"), to deal     #
# in the Software without restriction, including without limitation the rights      #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         #
# copies of the Software, and to permit persons to whom the Software is             #
# furnished to do so, subject to the following conditions:                          #
#                                                                                   #
# The above copyright notice and this permission notice shall be included in all    #
# copies or substantial portions of the Software.                                   #
#                                                                                   #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     #
# SOFTWARE.                                                                         #
# ================================================================================= #

# ================================================ #
#     Build Engine for Python Extension C-Libs     #
# ================================================ #

# use the Ubuntu 18.04 LTS image as base
FROM ubuntu:18.04

# install build dependencies: c/c++ pyhton3 numpy
RUN apt-get update && \
    apt-get install -y build-essential python3 python3-dev python3-pip && \
    pip3 install numpy asserts && \
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
