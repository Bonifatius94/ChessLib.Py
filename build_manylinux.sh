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

# ===========================================================================
#               M A N Y L I N U X    W H E E L   B U I L D
# ===========================================================================

# expected usage:
# 0) clone the source code to be built onto the build machine
# 1) mount the source code rootdir to the internal path /build
# 2) mount the wheel output directory to the internal path /output
# 3) call this script with a script argument indicating the python version
#    you want to build a manylinux wheel for (e.g. version cp36-cp36m)

# call this script like in following example (using the manylinux1 build env):
# docker run -v $PWD/dist:/output -v $PWD:/build -it \
#        quay.io/pypa/manylinux1_x86_64 \
#        bash /build/build_manylinux.sh cp36-cp36m

# ===========================================================================

# retrieve the python version to build for as script argument
PYTHON_VERSION=$1

# define the manylinux build toolchain path for the given python
# version that the chesslib wheels should be built for
PYTHON_ROOT=/opt/python/$PYTHON_VERSION/bin

# install the chesslib build dependencies (according to requirements.txt file)
$PYTHON_ROOT/pip install -r /build/requirements.txt

# build the chesslib wheel for the given python version
cd /build && $PYTHON_ROOT/pip wheel /build -w /output-out

echo 'created intermediate wheel file'
ls /output-out

# use the auditwheel tool to convert the wheel into a generic manywheel
# artifact that gets written to the /output directory 
auditwheel repair /output-out/chesslib*$PYTHON_VERSION*.whl -w /output

echo 'created manylinux wheel file'
ls /output/*$PYTHON_VERSION*

# TODO: add some kind of verification that the build was successful
