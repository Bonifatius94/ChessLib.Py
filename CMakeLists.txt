# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required(VERSION 3.8)

project("ChessLib")

# define paths of sub-projects
set(ChessLib "ChessLib")
set(ChessLib_PyMod "ChessLib.PyMod")

# include sub-projects
add_subdirectory(${ChessLib})
add_subdirectory(${ChessLib_PyMod})