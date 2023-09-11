cmake_minimum_required(VERSION 3.24)
project(muc LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)
flex_target(lexer lexer.l "${CMAKE_CURRENT_BINARY_DIR}/lex.cpp")
bison_target(parser parser.y "${CMAKE_CURRENT_BINARY_DIR}/parser.cpp" VERBOSE)

set(MU_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(MU_BINARY_DIR ${PROJECT_BINARY_DIR})

include_directories(${MU_SOURCE_DIR})
include_directories(${MU_BINARY_DIR})

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wno-sign-compare \
                     -Wno-unused-parameter -Wno-unused-function")
add_executable(muc main.cpp yyvaltype.cpp
                            "${CMAKE_CURRENT_BINARY_DIR}/lex.cpp"
                            "${CMAKE_CURRENT_BINARY_DIR}/parser.cpp")

