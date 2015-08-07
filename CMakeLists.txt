cmake_minimum_required (VERSION 2.6)
project (CUDA-CNN)

# The version number.
set (CUDA-CNN_VERSION_MAJOR 2)
set (CUDA-CNN_VERSION_MINOR 0)

#    CMake Scripts dir
#set(CMAKE_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/CMakeScripts)

#CMake module path for custom module finding
#set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SCRIPT_DIR})

find_package (CUDA REQUIRED)
include_directories (${CUDA_INCLUDE_DIRS})
#include_directories (${CUDA_SDK_ROOT_DIR})
MESSAGE("${CUDA_INCLUDE_DIRS}/../samples/common/inc")
include_directories ("${CUDA_INCLUDE_DIRS}/../samples/common/inc")
#/usr/local/cuda/NVIDIA_CUDA-7.0_Samples/common/inc")
set(CUDA_SEPARABLE_COMPILATION ON)

# opencv
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OPENCV_INCLUDE_DIR} )
set(OpenCV_LIBRARIES ${OpenCV_LIBS})


include_directories(${PROJECT_SOURCE_DIR})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -gencode arch=compute_20,code=sm_20
  -gencode arch=compute_20,code=sm_21
  -gencode arch=compute_30,code=sm_30
  -gencode arch=compute_35,code=sm_35
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_50,code=compute_50
  -rdc=true
  )

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  )

MESSAGE(${CUDA_NVCC_FLAGS})

#add_library(${EXAMPLE_NAME}.o OBJECT ${source})
#set(EXAMPLE_OBJ_LIB $<TARGET_OBJECTS:${EXAMPLE_NAME}.o>)


file(GLOB CUDA_CNN_CU_SOURCES  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
    "readData/*" 
    "layers/*"
    "dataAugmentation/*" 
    "common/*" 
    "main.cpp"
    "net.*")
    #  EXCLUDE_FROM_ALL "build/*")

cuda_add_executable(CUDA-CNN ${CUDA_CNN_CU_SOURCES})
#cuda_add_library(CUDA-CNN STATIC ${CUDA_CNN_CPP_SOURCES})
target_link_libraries(CUDA-CNN
  ${OpenCV_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDA_LIBRARIES}
  )
CUDA_ADD_CUBLAS_TO_TARGET(CUDA-CNN)

#add_subdirectory(net)
