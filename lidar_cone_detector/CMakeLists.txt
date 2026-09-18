cmake_minimum_required(VERSION 3.8)
project(lidar_cone_detector LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Orin NX 是 Ampere sm_87；本机 Ada 也常用于先编译检查，所以 x86 默认同时编译 87/89。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CUDA_ARCHITECTURES 87)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 87 89)
  endif()
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_default_runtime REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_path(NVINFER_INCLUDE_DIR NvInfer.h
  PATHS
    /usr/include/${CMAKE_LIBRARY_ARCHITECTURE}
    /usr/include
    /usr/local/cuda/include
)
find_library(NVINFER_LIBRARY nvinfer REQUIRED)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin REQUIRED)
if(NOT NVINFER_INCLUDE_DIR)
  message(FATAL_ERROR "NvInfer.h not found. Please install TensorRT C++ headers.")
endif()

# 1. 生成自定义消息
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ThreeDCone.msg"
  "msg/ThreeDConeArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# 2. C++/CUDA TensorRT 生产节点，支持 FP16 / INT8 / 混合精度 engine。
add_executable(ros2_infer_trt_cuda
  src/pointpillars_node.cpp
  src/trt_engine.cpp
  src/postprocess.cpp
  src/cuda_voxelizer.cu
  src/cuda/pillar_vfe.cu
  src/cuda/scatter_bev.cu
  src/cuda/gpu_postprocess.cu
)
target_include_directories(ros2_infer_trt_cuda PRIVATE
  include
  ${NVINFER_INCLUDE_DIR}
)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(ros2_infer_trt_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
  )
endif()
ament_target_dependencies(ros2_infer_trt_cuda
  rclcpp
  ament_index_cpp
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
)
target_link_libraries(ros2_infer_trt_cuda
  CUDA::cudart
  ${NVINFER_LIBRARY}
  ${NVINFER_PLUGIN_LIBRARY}
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(ros2_infer_trt_cuda ${cpp_typesupport_target})

install(TARGETS
  ros2_infer_trt_cuda
  DESTINATION lib/${PROJECT_NAME}
)

# 3. 安装运行所需资源。
install(DIRECTORY
  models
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
