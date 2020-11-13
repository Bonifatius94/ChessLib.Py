
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


if __name__ == "__main__":

    # generate a version and write it to 'version' file
    version = generate_package_version()
    f = open("version", "w")
    f.write(version)
    f.close()
