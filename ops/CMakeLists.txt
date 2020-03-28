cmake_minimum_required(VERSION 3.0)
project(oneflow_yolov3)
execute_process (
    COMMAND bash -c "python3 -c 'import oneflow; print(oneflow.sysconfig.get_include(), end=\"\")'"
    OUTPUT_VARIABLE ONEFLOW_INCLUDE_PATH
)

execute_process (
    COMMAND bash -c "python3 -c 'import oneflow; print(\" \".join(oneflow.sysconfig.get_compile_flags()), end=\"\")'"
    OUTPUT_VARIABLE ONEFLOW_COMPILE_FLAGS
)

execute_process (
    COMMAND bash -c "python3 -c 'import oneflow; print(oneflow.sysconfig.get_lib(), end=\"\")'"
    OUTPUT_VARIABLE ONEFLOW_LIB_PATH
)

find_package(CUDA REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CUDA_HOST_COMPILER g++)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -Xcompiler -Wextra --disable-warnings -DWITH_CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ONEFLOW_COMPILE_FLAGS} -O3 -g -std=c++11 -Wall -Wno-sign-compare -Wno-unused-function -fPIC")
set(CUDA_VERBOSE_BUILD OFF)

file(GLOB_RECURSE SRC "*.cpp")
file(GLOB_RECURSE HDR "*.h")
file(GLOB_RECURSE CUDASRC "*.cu")
file(GLOB_RECURSE CUDAHDR "*.cuh")
list(APPEND SRC ${HDR})
list(APPEND SRC ${CUDASRC})
list(APPEND SRC ${CUDAHDR})

cuda_add_library(${PROJECT_NAME} SHARED ${SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ../../yolov3)
target_include_directories(${PROJECT_NAME} PUBLIC ${ONEFLOW_INCLUDE_PATH})
target_link_libraries(${PROJECT_NAME} ${ONEFLOW_LIB_PATH}/_oneflow_internal.so)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/ops/darknet_dependencies/include)
target_link_libraries(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/ops/darknet_dependencies/libdarknet.so)


export(PACKAGE ${PROJECT_NAME})
export(TARGETS ${PROJECT_NAME}
FILE "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
