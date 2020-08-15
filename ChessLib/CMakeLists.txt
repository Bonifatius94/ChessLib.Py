# CMakeList.txt : CMake project for ChessLib, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# define the chesslib api library
add_library(ChessLib 
    "chessbitboard.c" "chessbitboard.h" 
    "chesspiece.h" "chesspiece.c" 
    "chessposition.h" "chessposition.c" 
    "chesslibmodule.h" "chesslibmodule.c" 
    "chessdraw.h" "chessdraw.c" 
    "chessdrawgen.h" "chessdrawgen.c" 
    "chesspieceatpos.h" "chesspieceatpos.c"
    "chesstypes.h")

# include and link python tools
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(ChessLib ${PYTHON_LIBRARIES})

# TODO: add unit tests