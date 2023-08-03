project(pacc)

cmake_minimum_required(VERSION 3.24)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

flex_target(lexer lexer.l "${CMAKE_CURRENT_BINARY_DIR}/lex.cpp")
bison_target(parser parser.y "${CMAKE_CURRENT_BINARY_DIR}/parser.cpp")

add_executable(pacc main.cpp "${CMAKE_CURRENT_BINARY_DIR}/lex.cpp"
                             "${CMAKE_CURRENT_BINARY_DIR}/parser.cpp")

set(CMAKE_CXX_FLAGS "-std=c++17 -Wc++17-extensions -I ${CMAKE_SOURCE_DIR}")
