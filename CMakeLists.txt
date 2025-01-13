cmake_minimum_required(VERSION 3.20)

project(Trt_plugin_example LANGUAGES CXX CUDA)

# 设置CUDA相关选项
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

# TensorRT路径
set(TRT_ROOT_DIR "D:/gwx/code/vs2022/tensorrt/TensorRT-10.6.0.26")

# 添加CUDA include路径
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(${PROJECT_NAME})

target_sources(
    ${PROJECT_NAME}
    PUBLIC
    source/plugin/cuda_utils.cuh
    source/plugin/cuda_impl.h
    source/plugin/FpsamplePlugin.h

    PRIVATE
    source/plugin/fpsample_gpu.cu
    source/plugin/FpsamplePlugin.cpp
    source/main.cpp
)

# 设置CUDA文件的编译选项
set_source_files_properties(
    source/fpsample_gpu.cu 
    PROPERTIES
    CUDA_SOURCE_PROPERTY_FORMAT OBJ
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${TRT_ROOT_DIR}/include
    ${TRT_ROOT_DIR}/include/common
    ${CMAKE_CURRENT_SOURCE_DIR}/source  # 添加source目录到include路径
)

# 添加CUDA运行时库
# find_package(CUDA REQUIRED)
target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    ${TRT_ROOT_DIR}/lib/nvinfer_10.lib
    ${TRT_ROOT_DIR}/lib/nvinfer_dispatch_10.lib
    ${TRT_ROOT_DIR}/lib/nvinfer_lean_10.lib
    ${TRT_ROOT_DIR}/lib/nvinfer_plugin_10.lib
    ${TRT_ROOT_DIR}/lib/nvinfer_vc_plugin_10.lib
    ${TRT_ROOT_DIR}/lib/nvonnxparser_10.lib
    ${CUDA_LIBRARIES}  # 添加CUDA运行时库
)

# 复制指定的DLL文件
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${TRT_ROOT_DIR}/lib/nvinfer_10.dll"
        "${TRT_ROOT_DIR}/lib/nvinfer_dispatch_10.dll"
        "${TRT_ROOT_DIR}/lib/nvinfer_lean_10.dll"
        "${TRT_ROOT_DIR}/lib/nvinfer_plugin_10.dll"
        "${TRT_ROOT_DIR}/lib/nvinfer_vc_plugin_10.dll"
        "${TRT_ROOT_DIR}/lib/nvonnxparser_10.dll"
        "${TRT_ROOT_DIR}/lib/nvinfer_builder_resource_10.dll"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

# 设置MSVC特定选项
if(MSVC)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        LINK_FLAGS "/NODEFAULTLIB:LIBCMT"
    )
endif()