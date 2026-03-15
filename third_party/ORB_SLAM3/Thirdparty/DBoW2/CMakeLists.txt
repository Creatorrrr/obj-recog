cmake_minimum_required(VERSION 3.5)
project(DBoW2)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
endif()

set(HDRS_DBOW2
  DBoW2/BowVector.h
  DBoW2/FORB.h 
  DBoW2/FClass.h       
  DBoW2/FeatureVector.h
  DBoW2/ScoringObject.h   
  DBoW2/TemplatedVocabulary.h)
set(SRCS_DBOW2
  DBoW2/BowVector.cpp
  DBoW2/FORB.cpp      
  DBoW2/FeatureVector.cpp
  DBoW2/ScoringObject.cpp)

set(HDRS_DUTILS
  DUtils/Random.h
  DUtils/Timestamp.h)
set(SRCS_DUTILS
  DUtils/Random.cpp
  DUtils/Timestamp.cpp)


find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 3.0 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 3.0 not found.")
   endif()
endif()
find_package(Boost CONFIG REQUIRED)
if(TARGET Boost::headers)
  get_target_property(BOOST_INCLUDE_DIRS Boost::headers INTERFACE_INCLUDE_DIRECTORIES)
elseif(TARGET Boost::boost)
  get_target_property(BOOST_INCLUDE_DIRS Boost::boost INTERFACE_INCLUDE_DIRECTORIES)
endif()

set(DBOW2_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${DBOW2_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${DBOW2_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DBOW2_OUTPUT_DIRECTORY})
foreach(config Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${config}" config_upper)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper} ${DBOW2_OUTPUT_DIRECTORY})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper} ${DBOW2_OUTPUT_DIRECTORY})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${DBOW2_OUTPUT_DIRECTORY})
endforeach()

if(MSVC)
  set(DBOW2_LIB_TYPE STATIC)
else()
  set(DBOW2_LIB_TYPE SHARED)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${BOOST_INCLUDE_DIRS})
add_library(DBoW2 ${DBOW2_LIB_TYPE} ${SRCS_DBOW2} ${SRCS_DUTILS})
target_link_libraries(DBoW2 ${OpenCV_LIBS})
