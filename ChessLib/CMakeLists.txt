# CMakeList.txt : CMake project for ChessLib, include source and define
# project specific logic here.
#
cmake_minimum_required(VERSION 3.8)

# define the chesslib api library
include_directories("include")
add_library(ChessLib 
    "src/ChessBoard.c" 
    "src/chesspiece.c" 
    "src/chessposition.c" 
    "src/chessdraw.c" 
    "src/chessdrawgen.c" 
    "src/chesspieceatpos.c")

# TODO: add unit tests