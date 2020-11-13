
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

# =============================================
#      SETUP CHESSLIB PYTHON EXTENSION
# =============================================


# import python C-lib extension tools
from setuptools import setup, Extension
import numpy as np
import sys, os, io
# TODO: try the alternative scikit build tools allowing for the integration of CMake builds


# define configurable variables
PACKAGE_NAME = "chesslib"
DESCRIPTION = "Python3 library for efficient chess draw-gen functions"
PROJECT_URL = "https://github.com/Bonifatius94/ChessLib.Py"
AUTHOR = "Marco Tröster"
AUTHOR_EMAIL = "marco@troester-gmbh.de"
PYTHON_VERSION = '>=3.0.0'
DEPENDENCIES = ['numpy']

# define semantic major and minor version (this may be changed manually)
__MAJOR_VERSION__ = 1
__MINOR_VERSION__ = 0


def generate_package_version():
    """
    Generate a unique package version of the format
    'major_version.minor_version.build'.
    """

    unique_build_no = generate_unique_build_number()
    return '{}.{}.{}'.format(__MAJOR_VERSION__, __MINOR_VERSION__, unique_build_no)


def generate_unique_build_number():
    """
    Unique build number generator function providing
    growing build numbers using seconds/2 since millenium formula.
    """

    # import the standard datetime lib
    from datetime import datetime
    from datetime import timezone

    # create UTC timestamp of now
    utc_now = datetime.now(timezone.utc)

    # create UTC timestamp millenium 2000
    utc_millenium = datetime(2000, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)

    # determine the total seconds / 2 since millenium
    return round((utc_now - utc_millenium).total_seconds() / 2)


def load_readme_description():

    here = os.path.abspath(os.path.dirname(__file__))
    try:
        with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
            long_description = '\n' + f.read()
    except FileNotFoundError:
        long_description = DESCRIPTION


# declare main function
def main():

    # generate a unique package version
    version = generate_package_version()

    # define source files to compile as python C-lib extension module
    source_files = [
        "chesslib/src/chesslibmodule.c",
        "chesslib/src/chessboard.c",
        "chesslib/src/chesspiece.c",
        "chesslib/src/chessposition.c",
        "chesslib/src/chessdraw.c",
        "chesslib/src/chessdrawgen.c",
        "chesslib/src/chesspieceatpos.c",
        "chesslib/src/chessgamestate.c"
    ]

    # define extension module settings (for cross-plattform builds)
    chesslib_module = Extension("chesslib",
                                include_dirs = ["chesslib/include", np.get_include()],
                                sources = source_files,
                                define_macros = [("CHESSLIB_PACKAGE_VERSION", '"{}"'.format(version))],
                                language = 'c')

    # setup python extension module (compilation, packaging, deployment, ...)
    setup(name = PACKAGE_NAME,
          version = version,
          description = DESCRIPTION,
          long_description = load_readme_description(),
          long_description_content_type = 'text/markdown',
          url = PROJECT_URL,
          author = AUTHOR,
          author_email = AUTHOR_EMAIL,
          python_requires = PYTHON_VERSION,
          install_requires = DEPENDENCIES,
          ext_modules = [chesslib_module],
          setup_requires=['wheel'])


# invoke main function
if __name__ == "__main__":
    main()


# =============================================
#         Marco Tröster, 2020-11-13
# =============================================
