
# =============================================
#      SETUP CHESSLIB PYTHON EXTENSION
# =============================================


# import python C-lib extension tools
from numpy.distutils.core import setup, Extension
# TODO: try the alternative scikit build tools allowing for the integration of CMake builds


# declare main function
def main():

    # define source files to compile as python C-lib extension module
    source_files = [
        "src/chesslibmodule.c",
        "src/chessboard.c",
        "src/chesspiece.c",
        "src/chessposition.c",
        "src/chessdraw.c",
        "src/chessdrawgen.c",
        "src/chesspieceatpos.c"
    ]

    # define extension module settings (for cross-plattform builds)
    chesslib_module = Extension("chesslib", 
                                include_dirs = ["include"],
                                sources = source_files,
                                language = 'c')

    # setup python extension module (compilation, packaging, deployment, ...)
    setup(name = "chesslib",
          version = "1.0.0",
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
