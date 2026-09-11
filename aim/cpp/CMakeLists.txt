cmake_minimum_required(VERSION 3.18)
project(aim_onboard LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)

set(_trt_include_hints
    "$ENV{TENSORRT_ROOT}/include"
    "${TENSORRT_ROOT}/include"
    /usr/include/x86_64-linux-gnu
    /usr/include
    /usr/local/tensorrt/include
    /home/ubuntu/yolov5_fourpoints/trt_cpp/third_party/trt_oss/include
)
set(_trt_lib_hints
    "$ENV{TENSORRT_ROOT}/lib"
    "${TENSORRT_ROOT}/lib"
    /usr/lib/x86_64-linux-gnu
    /usr/local/tensorrt/lib
    /home/ubuntu/yolov5_fourpoints/venv/lib/python3.12/site-packages/tensorrt_libs
)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${_trt_include_hints})
find_library(TENSORRT_NVINFER NAMES nvinfer nvinfer.so.11 HINTS ${_trt_lib_hints})
find_library(TENSORRT_NVONNXPARSER NAMES nvonnxparser nvonnxparser.so.11 HINTS ${_trt_lib_hints})
if(NOT TENSORRT_NVINFER)
    find_file(TENSORRT_NVINFER NAMES libnvinfer.so.11 libnvinfer.so.10 libnvinfer.so.8
              HINTS ${_trt_lib_hints})
endif()
if(NOT TENSORRT_NVONNXPARSER)
    find_file(TENSORRT_NVONNXPARSER NAMES libnvonnxparser.so.11 libnvonnxparser.so.10 libnvonnxparser.so.8
              HINTS ${_trt_lib_hints})
endif()

if(NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "NvInfer.h not found. Set TENSORRT_ROOT or TENSORRT_INCLUDE_DIR.")
endif()
if(NOT TENSORRT_NVINFER OR NOT TENSORRT_NVONNXPARSER)
    message(FATAL_ERROR "TensorRT libraries not found. Set TENSORRT_ROOT or TENSORRT_LIB_DIR.")
endif()

get_filename_component(TENSORRT_LIB_DIR "${TENSORRT_NVINFER}" DIRECTORY)
message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer: ${TENSORRT_NVINFER}")
message(STATUS "CUDA include: ${CUDAToolkit_INCLUDE_DIRS}")

add_executable(camera_example camera_example.cpp trtengine.cpp ekf.cpp)
target_include_directories(camera_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(camera_example PRIVATE
    ${OpenCV_LIBS}
    ${TENSORRT_NVINFER}
    ${TENSORRT_NVONNXPARSER}
    CUDA::cudart
    Eigen3::Eigen
)
if(UNIX AND NOT APPLE)
    target_link_libraries(camera_example PRIVATE rt)
endif()
set_target_properties(camera_example PROPERTIES
    BUILD_RPATH "${TENSORRT_LIB_DIR};${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${TENSORRT_LIB_DIR};${CUDAToolkit_LIBRARY_DIR}"
)

file(CREATE_LINK
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_SOURCE_DIR}/compile_commands.json"
    SYMBOLIC)
