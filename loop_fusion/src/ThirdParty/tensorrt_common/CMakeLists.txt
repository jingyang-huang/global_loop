cmake_minimum_required(VERSION 3.6)
project(tensorrt_common)

set(CMAKE_CXX_STANDARD 17)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_library(tensorrt_common SHARED src/logger.cpp src/sampleOptions.cpp src/sampleUtils.cpp ) #src/sampleEngines.cpp
# add_library(tensorrtbuffer SHARED src/logger.cpp src/sample_utils.cpp)
