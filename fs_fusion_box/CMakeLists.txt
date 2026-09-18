cmake_minimum_required(VERSION 3.8)
project(fs_fusion_box)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ==========================================
# 1. 查找依赖
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(message_filters REQUIRED)
find_package(drd25_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cone_interfaces REQUIRED)  # 必须找到
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(lidar_cone_detector REQUIRED)

# ==========================================
# 2. 全局头文件包含
# ==========================================
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
)

# ==========================================
# 3. 编译核心库 (fs_fusion_box_lib)
# ==========================================
add_library(fs_fusion_box_lib 
  src/fs_fusion_box_math.cpp
  src/visualizer.cpp
)

# 链接库
target_link_libraries(fs_fusion_box_lib
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

# [核心修复] 补上 cone_interfaces
ament_target_dependencies(fs_fusion_box_lib
  rclcpp
  rcl_interfaces
  vision_msgs
  drd25_msgs
  sensor_msgs
  cv_bridge
  Eigen3
  lidar_cone_detector
  visualization_msgs
  cone_interfaces  # <--- 之前漏了这一行！导致 visualizer.hpp 找不到头文件
)

# ==========================================
# 4. 编译节点 (fusion_box_node)
# ==========================================
add_executable(fusion_box_node 
  src/fs_fusion_box_node.cpp
)

# 链接核心库
target_link_libraries(fusion_box_node 
  fs_fusion_box_lib
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

ament_target_dependencies(fusion_box_node
  rclcpp
  rcl_interfaces
  std_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
  visualization_msgs
  cv_bridge
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  message_filters
  drd25_msgs
  lidar_cone_detector
  cone_interfaces 
)

# ==========================================
# 5. 安装规则
# ==========================================
install(TARGETS 
  fs_fusion_box_lib
  fusion_box_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)



ament_package()
