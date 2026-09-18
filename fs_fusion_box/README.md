fs_fusion_box: FS Driverless 感知融合功能包

fs_fusion_box 是无人赛车感知系统的核心融合节点。它负责接收来自激光雷达 (LiDAR) 的 3D 锥筒聚类结果 和来自相机 (YOLO) 的 2D 识别框，通过时空同步与多模态融合算法，输出带有颜色分类和精确三维坐标的锥筒地图。

# fs_fusion_box: FS Driverless 感知融合功能包

**fs_fusion_box** 是无人赛车感知系统的核心融合节点。它负责接收来自激光雷达 (LiDAR) 的 **3D 锥筒聚类结果** 和来自相机 (YOLO) 的 **2D 识别框**，通过时空同步与多模态融合算法，输出带有颜色分类和精确三维坐标的锥筒地图。



## 1. 核心功能 (Features)

* **多模态融合**：将 3D 几何信息（LiDAR）与 2D 语义颜色信息（Camera）进行匹配。
* **双适配器架构**：内部集成了 Adapter 层，直接支持自定义消息类型 (`cone_interfaces` & `test_cone_segmentation`)，无需修改核心算法库。
* **时空同步**：使用 `ApproximateTime` 策略自动对齐雷达与相机的时间戳。
* **单目补漏**：针对雷达漏检但在相机视野内的锥筒，基于单目测距原理反推 3D 坐标。
* **可视化调试**：提供 Rviz MarkerArray 接口，实时显示融合后的锥筒位置与颜色。

## 2. 依赖项 (Dependencies)



在编译此包之前，请确保你的工作空间 (`src`) 中包含以下自定义消息包，并且系统已安装必要的库。

### 2.1 自定义消息包 (必须存在于工作空间中)
* **`drd25_msgs`**: 定义了最终输出的 `Map` 和 `Cone` 消息。
* **`test_cone_segmentation`**: 提供 3D 雷达聚类输入 (`ThreeDConeArray`)。
* **`cone_interfaces`**: 提供 2D YOLO 检测输入 (`ConeArray`)。

### 2.2 系统依赖 (System Libraries)
* ROS 2 (Humble/Foxy)
* OpenCV (`libopencv-dev`)
* Eigen3
* PCL (Point Cloud Library)

### 2.3 ROS 2 依赖包
安装命令：
```bash
sudo apt update
sudo apt install ros-$ROS_DISTRO-vision-msgs \
                 ros-$ROS_DISTRO-cv-bridge \
                 ros-$ROS_DISTRO-message-filters \
                 ros-$ROS_DISTRO-image-transport \
                 ros-$ROS_DISTRO-tf2-eigen





可视化坐标系为：hesai_lidar