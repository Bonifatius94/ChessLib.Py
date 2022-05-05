
cmake_minimum_required(VERSION 3.9.4)
project(chesslib LANGUAGES C)

file(GLOB sources "src/*.c")
add_library(chesslib ${sources})
target_include_directories(chesslib PRIVATE include)
set_property(TARGET chesslib PROPERTY C_STANDARD 90)

add_subdirectory(test)
enable_testing()
add_test(NAME UnitTests COMMAND UnitTests)
