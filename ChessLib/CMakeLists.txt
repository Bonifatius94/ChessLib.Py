# =========================================
#           PROJECT    CHESSLIB
# =========================================

# set cmake tool version
cmake_minimum_required(VERSION 3.8)

# declare the project
project("ChessLib")

# include local header files
include_directories("include")

# include python C-lib tools (with NumPy)
find_package(Python3 COMPONENTS Interpreter Development NumPy)
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

# set compilter flags (C89 standard)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -pedantic")
endif()

# create library with python module code
set(ChessLib "ChessLib")
add_library(${ChessLib} SHARED
    "src/chesslibmodule.c"
    "src/chessboard.c"
    "src/chesspiece.c"
    "src/chessposition.c"
    "src/chessdraw.c"
    "src/chessdrawgen.c"
    "src/chesspieceatpos.c")
target_link_libraries(${ChessLib}  ${Python3_LIBRARIES})
# TODO: disable CMake debug mode, only use release by default (otherwise the build will run into an error with missing python debug components)
# TODO: compile as shared-lib using the SHARED keyword in add_library definition

# install the library properly, so it can be used
install(TARGETS ${ChessLib}
        LIBRARY
          DESTINATION lib
          COMPONENT Libraries
          NAMELINK_COMPONENT Development
        PUBLIC_HEADER
          DESTINATION include
          COMPONENT Development)

# TODO: add unit tests
