cmake_minimum_required(VERSION 3.20)
project(obj_recog_orbslam3_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
foreach(config Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${config}" config_upper)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_BINARY_DIR})
endforeach()

if(NOT DEFINED ORB_SLAM3_ROOT)
  message(FATAL_ERROR "ORB_SLAM3_ROOT must point to a local ORB_SLAM3 checkout")
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(orbslam3_bridge bridge_main.cpp)
target_compile_definitions(
  orbslam3_bridge
  PRIVATE
    COMPILEDWITHC11
    ORB_SLAM3_HEADLESS_VIEWER=1
)
target_include_directories(
  orbslam3_bridge
  PRIVATE
    ${ORB_SLAM3_ROOT}/include
    ${ORB_SLAM3_ROOT}/include/CameraModels
    ${ORB_SLAM3_ROOT}
    ${ORB_SLAM3_ROOT}/Thirdparty/Sophus
)

target_link_directories(
  orbslam3_bridge
  PRIVATE
    ${ORB_SLAM3_ROOT}/lib
)

target_link_libraries(
  orbslam3_bridge
  PRIVATE
    ORB_SLAM3
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
