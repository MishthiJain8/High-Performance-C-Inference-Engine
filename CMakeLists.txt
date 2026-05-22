cmake_minimum_required(VERSION 3.10)
project(HighPerformanceInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(NOT DEFINED ONNXRUNTIME_DIR)
  if(DEFINED ENV{ONNXRUNTIME_DIR})
    set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
  endif()
endif()

if(NOT ONNXRUNTIME_DIR)
  message(FATAL_ERROR "ONNXRUNTIME_DIR not set. Configure with -DONNXRUNTIME_DIR=/path/to/onnxruntime")
endif()

message(STATUS "Using ONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}")

set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_DIR}/lib)

find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h HINTS ${ONNXRUNTIME_INCLUDE_DIR})
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime onnxruntime_providers_shared HINTS ${ONNXRUNTIME_LIB_DIR})

if(NOT ONNXRUNTIME_INCLUDE)
  message(FATAL_ERROR "Could not find onnxruntime headers in ${ONNXRUNTIME_INCLUDE_DIR}")
endif()
if(NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "Could not find onnxruntime library in ${ONNXRUNTIME_LIB_DIR}")
endif()

add_executable(inference
  src/main.cpp
)

target_include_directories(inference PRIVATE ${ONNXRUNTIME_INCLUDE})
target_compile_options(inference PRIVATE -O3 -march=native -fno-omit-frame-pointer)

target_link_libraries(inference PRIVATE ${ONNXRUNTIME_LIBRARY} pthread dl)

set_target_properties(inference PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
  INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
)
