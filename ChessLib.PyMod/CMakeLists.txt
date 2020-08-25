# CMakeList.txt : CMake project for ChessLib, include source and define
# project specific logic here.
#
cmake_minimum_required(VERSION 3.8)

# include python c-lib tools
#find_package(PythonLibs REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy)
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

# include the chesslib api
include_directories("include" "../ChessLib/include")

# create library with python module code
add_library(ChessLib_PyMod "src/chesslibmodule.c")
target_link_libraries(ChessLib_PyMod ${Python3_LIBRARIES})

# TODO: add unit tests (preferrably in python)