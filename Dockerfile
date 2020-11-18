
# use Linux Alpine / Slim with pre-installed Python3 as base image
# TODO: figure out which base image works best
#FROM python:3-alpine
FROM python:3-slim
#FROM ubuntu:18.04

# pass the PyPi package version to be shipped with this image
ARG PYPI_PACKAGE_VERSION
RUN test -n "$PYPI_PACKAGE_VERSION"

RUN pip --version

# install the PyPi 'chesslib' package of the given version
RUN pip install numpy chesslib==$PYPI_PACKAGE_VERSION

# short test if the PyPi package can be imported with python
RUN if ! [ $(python -c "exec(\"import chesslib\nprint(chesslib.version)\")") == $PYPI_PACKAGE_VERSION ]; then exit -1 ; fi

# info: chesslib package is now ready to use ...
