
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

# ================================================================================= #
# This module defines global Python lib variables and functions to retrieve the     #
# current package version properly. Note that this file's package version is the    #
# single source of truth for the entire project, so don't hard-code the package     #
# version anywhere else!!!                                                          #
#                                                                                   #
# Note: The version provided by this module changes after each module import.       #
#       Therefore use the pip3 package info to retrieve the package's version       #
#       after the package creation.                                                 #
# ================================================================================= #


# import the standard datetime lib
from datetime import datetime
from datetime import timezone


# define semantic major and minor version (this may be changed manually)
__MAJOR_VERSION__ = 1
__MINOR_VERSION__ = 0

# init a unique package version remaining the same until the module gets reloaded
__BUILD_NO__ = generate_unique_build_number()
__PACKAGE_VERSION__ = '{}.{}.{}'.format(__MAJOR_VERSION__, __MINOR_VERSION__, __BUILD_NO__)


def update_version_file():
    """
    Update the current package version script file defining 
    the global __version__ string variable.
    """
    
    # write the __version__ variable to the __version__.py script file (overwrite if file exists)
    version_file = open("__version__.py", "w")
    version_file.write('__version__ = "{}"'.format(__PACKAGE_VERSION__))
    version_file.close()


def generate_unique_build_number():
    """
    Unique build number generator function providing 
    growing build numbers using seconds/2 since millenium formula.
    """
    
    # create UTC timestamp of now
    utc_now = datetime.now(timezone.utc)
    
    # create UTC timestamp millenium 2000
    utc_millenium = datetime.datetime(2000, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
    
    # determine the total seconds / 2 since millenium
    return (utc_now - utc_millenium).total_seconds() / 2
    
    
