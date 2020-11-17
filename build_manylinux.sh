
# ===========================================================================
# info: this script is supposed to be run inside the manylinux1 docker image
# ===========================================================================

# define build arguments
PYTHON_VERSION=$1
PYTHON_ROOT=/opt/python/$PYTHON_VERSION/bin

# install build dependencies
$PYTHON_ROOT/pip install setuptools twine numpy

# build manylinux wheels (expects /output directory to exist)
cd /build
$PYTHON_ROOT/pip wheel /build -w /output
auditwheel repair /output/chesslib*whl -w /output
