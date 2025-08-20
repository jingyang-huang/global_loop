cmake_minimum_required(VERSION 2.8.3)
project(loop_fusion)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++17")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
add_definitions(-w)
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

#[Dependency] tensorrt
set(tensorrt_dir "/home/hjy/Library/TensorRT-8.5.2.2.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.2.2")
include_directories(${tensorrt_dir}/include)
link_directories(${tensorrt_dir}/lib)


#[Dependency] cuda
set(CUDA_DIR  "/usr/local/cuda-11.4")
find_package(CUDA REQUIRED)
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)

add_subdirectory(${PROJECT_SOURCE_DIR}/src/ThirdParty/tensorrt_common)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/ThirdParty/DBoW2)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/ThirdParty/backward-cpp)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    camera_models
    cv_bridge
    roslib
    pcl_ros
    tf
    message_generation
    )

# find_package(OpenCV 4.2.0 EXACT REQUIRED PATHS /home/nv/Libs/opencv-4.2.0)
# [Dependency] OpenCV
find_package(OpenCV 4.2.0 REQUIRED)

# find_package(brisk 2 REQUIRED )
# set(Ceres_DIR "/home/ubuntu/Dependency/ceres-1.14.0")
# find_package(Ceres NO_DEFAULT_PATH PATHS ${PROJECT_SOURCE_DIR}/Thirdparty/ceres-solver/install/)
# [Dependency] Ceres0
find_package(Ceres REQUIRED)
# 输出ceres版本
message(STATUS "Ceres version: ${Ceres_VERSION}")

find_package(PCL REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3)


catkin_package(
 LIBRARIES ${PROJECT_NAME}_lib
 CATKIN_DEPENDS geometry_msgs image_transport nav_msgs roscpp rospy std_msgs
)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}  
    ${EIGEN3_INCLUDE_DIR} 
    ${CUDA_INCLUDE_DIRS}
    ${BRISK_INCLUDE_DIRS})
message(STATUS "Using system brisk. Found brisk at ${BRISK_INCLUDE_DIRS}.")


# add_library(brisk STATIC IMPORTED)
# add_library(agast STATIC IMPORTED)
# set(BRISK_LIBRARIES brisk agast)
# set_target_properties(brisk PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/lib/libbrisk.a)
# set_target_properties(agast PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/lib/libagast.a)
# add_dependencies(brisk brisk_external)
# add_dependencies(agast brisk_external)

add_library(loop_fusion_lib SHARED
    src/pose_graph.cpp
    src/keyframe.cpp
    src/feature_tracker.cpp
    src/feature_manager.cpp

    src/feature_detector.cpp
    src/super_glue.cpp
    src/super_point.cpp
    src/parameters.cpp
    src/point_matcher.cpp

    src/utils.cpp
    src/utility/CameraPoseVisualization.cpp
    src/utility/dls_pnp.cc
    src/utility/dls_impl.cc
    src/utility/utility.cpp
    src/database.cpp
    src/FSuperpoint.cpp

    # src/ThirdParty/DBoW2/BowVector.cpp
    # src/ThirdParty/DBoW2/FBrief.cpp
    # src/ThirdParty/DBoW2/FeatureVector.cpp
    # src/ThirdParty/DBoW2/QueryResults.cpp
    # src/ThirdParty/DBoW2/ScoringObject.cpp
    # src/ThirdParty/DBoW2/FSuperpoint.cpp
    # src/ThirdParty/DBoW2/database.cpp

    # src/ThirdParty/brisk/src/brisk-descriptor-extractor.cc
    # src/ThirdParty/brisk/src/brisk-feature-detector.cc
    src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp
    src/ThirdParty/DVision/BRIEF.cpp
    src/ThirdParty/VocabularyBinary.cpp

    )
# 没有这句找不到tensorrt和cuda
target_link_libraries(${PROJECT_NAME}_lib
    nvinfer
    nvonnxparser
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${PCL_LIBS} 
    ${CERES_LIBRARIES} 
    ${BRISK_LIBRARIES}
    tensorrt_common
    DBoW2
    -lboost_serialization
)

add_executable(ros_mapping_node src/ros_mapping_node.cpp ${BACKWARD_ENABLE})
target_link_libraries(ros_mapping_node loop_fusion_lib ${catkin_LIBRARIES} ) 
add_backward(ros_mapping_node)

add_executable(ros_reloc_node src/ros_reloc_node.cpp ${BACKWARD_ENABLE})
target_link_libraries(ros_reloc_node loop_fusion_lib ${catkin_LIBRARIES}) 
add_backward(ros_reloc_node)