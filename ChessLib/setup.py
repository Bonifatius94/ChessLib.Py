
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
# TODO: try the alternative scikit build tools allowing for the integration of CMake builds
from distutils.core import setup, Extension
import numpy as np
import sys
import package_version_gen


# declare main function
def main():
    
    # generate unique package version using package_version_gen.py script
    version = package_version_gen.__PACKAGE_VERSION__
    #package_version_gen.update_version_file()
    
    # define source files to compile as python C-lib extension module
    source_files = [
        "src/chesslibmodule.c",
        "src/chessboard.c",
        "src/chesspiece.c",
        "src/chessposition.c",
        "src/chessdraw.c",
        "src/chessdrawgen.c",
        "src/chesspieceatpos.c",
        "src/chessgamestate.c"
    ]

    # define extension module settings (for cross-plattform builds)
    chesslib_module = Extension("chesslib", 
                                include_dirs = ["include", np.get_include()],
                                sources = source_files,
                                language = 'c')

    # setup python extension module (compilation, packaging, deployment, ...)
    setup(name = "chesslib",
          version = version,
          description = "Python interface for efficient chess draw-gen C library functions",
          author = "Marco Tröster",
          author_email = "marco@troester-gmbh.de",
          ext_modules = [chesslib_module])


# invoke main function
if __name__ == "__main__":
    main()


# =============================================
#         Marco Tröster, 2020-08-29
# =============================================
