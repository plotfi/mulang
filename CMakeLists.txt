cmake_minimum_required(VERSION 3.24)
project(muc)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

flex_target(lexer lexer.l "${CMAKE_CURRENT_BINARY_DIR}/lex.cpp")
bison_target(parser parser.y "${CMAKE_CURRENT_BINARY_DIR}/parser.cpp" VERBOSE)

add_executable(muc main.cpp yyvaltype.cpp
                            "${CMAKE_CURRENT_BINARY_DIR}/lex.cpp"
                            "${CMAKE_CURRENT_BINARY_DIR}/parser.cpp")

set(CMAKE_CXX_FLAGS "-std=c++17 -Wc++17-extensions -Wall -Wextra -Werror -Wno-sign-compare -Wno-unused-parameter -Wno-unused-function -I ${CMAKE_SOURCE_DIR}")
