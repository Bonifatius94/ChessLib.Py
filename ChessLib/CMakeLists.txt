# CMakeList.txt : CMake project for ChessLib, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# define the chesslib api library
include_directories("include")
add_library(ChessLib 
    "src/ChessBoard.c" 
    "src/chesspiece.c" 
    "src/chessposition.c" 
    "src/chesslibmodule.c" 
    "src/chessdraw.c" 
    "src/chessdrawgen.cpp" 
    "src/chesspieceatpos.c")

#add_library(ChessLib 
#    "include/ChessBoard.h"   "src/ChessBoard.c" 
#    "include/chesspiece.h"      "src/chesspiece.c" 
#    "include/chessposition.h"   "src/chessposition.c" 
#    "include/chesslibmodule.h"  "src/chesslibmodule.c" 
#    "include/chessdraw.h"       "src/chessdraw.c" 
#    "include/chessdrawgen.h"    "src/chessdrawgen.c" 
#    "include/chesspieceatpos.h" "src/chesspieceatpos.c"
#    "include/chesstypes.h")

# include and link python tools
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(ChessLib ${PYTHON_LIBRARIES})

# TODO: add unit tests