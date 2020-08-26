# =========================================
#      SUB - PROJECT    CHESS LIB
# =========================================

# set cmake tool version
cmake_minimum_required(VERSION 3.8)

# set compilter flags (C89 standard)
if(MSVC) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -pedantic")
endif()

# include local header files
include_directories("include")

# include python C-lib tools (with NumPy)
find_package(Python3 COMPONENTS Interpreter Development NumPy)
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

# create library with python module code
set(ChessLib "ChessLib")
add_library(${ChessLib} 
    "src/chesslibmodule.c"
    "src/ChessBoard.c" 
    "src/chesspiece.c" 
    "src/chessposition.c" 
    "src/chessdraw.c" 
    "src/chessdrawgen.c" 
    "src/chesspieceatpos.c")

target_link_libraries(${ChessLib}  ${Python3_LIBRARIES})

# TODO: add unit tests