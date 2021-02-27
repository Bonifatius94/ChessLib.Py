
# ===========================================================================
# info: this script is supposed to be run inside the manylinux1 docker image
# ===========================================================================

# define build arguments
PYTHON_VERSION=$1
PYTHON_ROOT=/opt/python/$PYTHON_VERSION/bin

cd /build

# install build dependencies
$PYTHON_ROOT/pip install -r requirements.txt

# build manylinux wheels (expects /output directory to exist)
$PYTHON_ROOT/pip wheel /build -w /output-out
auditwheel repair /output-out/chesslib*$PYTHON_VERSION*.whl -w /output
