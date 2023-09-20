cmake_minimum_required(VERSION 3.24)
project(muc LANGUAGES CXX C)

set(LLVM_LINK_COMPONENTS
  Support
  )

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()

set(MU_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(MU_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${MU_SOURCE_DIR})
include_directories(${MU_BINARY_DIR})
include_directories(${MU_SOURCE_DIR}/include)
include_directories(${MU_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)

find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)
set(MU_PARSEGEN_SRC_DIR "${MU_SOURCE_DIR}/lib/Mu/Parser")

flex_target(lexer "${MU_PARSEGEN_SRC_DIR}/lexer.l"
                  "${CMAKE_CURRENT_BINARY_DIR}/lex.cpp")

bison_target(parser "${MU_PARSEGEN_SRC_DIR}/parser.y"
                    "${CMAKE_CURRENT_BINARY_DIR}/parser.cpp"
                    VERBOSE)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wno-sign-compare \
                     -Wno-unused-parameter -Wno-unused-function \
                     -Wno-ambiguous-reversed-operator")

add_executable(muc main.cpp yyvaltype.cpp
                   "${CMAKE_CURRENT_BINARY_DIR}/lex.cpp"
                   "${CMAKE_CURRENT_BINARY_DIR}/parser.cpp")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        MLIRMu
        )

llvm_update_compile_flags(muc)
target_link_libraries(muc
  PRIVATE
    MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms
    ${LIBS})
