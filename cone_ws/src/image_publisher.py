#!/usr/bin/env python3
"""
图片发布节点
从本地图片文件发布到 ROS2 话题，用于测试检测节点
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pathlib import Path
import time


class ImagePublisher(Node):
    def __init__(self, image_path, topic_name='/camera/image_raw', rate=1, max_count=None):
        super().__init__('image_publisher')
        
        self.bridge = CvBridge()
        self.image_path = image_path
        self.topic_name = topic_name
        self.rate = rate
        self.max_count = max_count
        self.image_paths = []
        self.image_index = 0
        self.ok = True
        
        # 检查图片文件
        path = Path(image_path)
        if not path.exists():
            self.get_logger().error(f'图片或目录不存在: {image_path}')
            self.ok = False
            return
        
        if path.is_dir():
            self.image_paths = sorted(
                [str(p) for p in path.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            )
            if not self.image_paths:
                self.get_logger().error(f'目录中没有图片: {image_path}')
                self.ok = False
                return
            self.get_logger().info(f'✓ 已加载目录: {image_path} (共 {len(self.image_paths)} 张)')
            first = cv2.imread(self.image_paths[0])
            if first is not None:
                self.get_logger().info(f'✓ 示例图片大小: {first.shape[1]}x{first.shape[0]}')
            self.image = None
        else:
            # 读取图片
            self.image = cv2.imread(image_path)
            if self.image is None:
                self.get_logger().error(f'无法读取图片: {image_path}')
                self.ok = False
                return
            self.get_logger().info(f'✓ 已加载图片: {image_path}')
            self.get_logger().info(f'✓ 图片大小: {self.image.shape[1]}x{self.image.shape[0]}')
        
        # 创建发布器
        self.pub = self.create_publisher(Image, topic_name, 10)
        
        # 创建定时器
        self.timer = self.create_timer(1.0 / rate, self.publish_image)
        self.count = 0


    def publish_image(self):
        """发布图片"""
        if self.image is None and not self.image_paths:
            return
        
        if self.image_paths:
            image_path = self.image_paths[self.image_index]
            image = cv2.imread(image_path)
            if image is None:
                self.get_logger().warn(f'无法读取图片: {image_path}')
                return
            self.image_index = (self.image_index + 1) % len(self.image_paths)
        else:
            image = self.image
        
        # 转换为 ROS Image 消息
        msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
        self.pub.publish(msg)
        
        self.count += 1
        if self.count % 10 == 0:
            self.get_logger().info(f'已发布 {self.count} 张图片到 {self.topic_name}')
        if self.max_count is not None and self.count >= self.max_count:
            self.get_logger().info(f'达到发布数量上限 {self.max_count}，自动退出')
            rclpy.shutdown()


def main(args=None):
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python3 image_publisher.py <image_path|dir> [topic_name] [rate] [count]")
        print("     image_path|dir: 图片文件路径或图片目录")
        print("     topic_name: 发布的话题名 (默认: /camera/image_raw)")
        print("     rate: 发布频率，Hz (默认: 1)")
        print("     count: 发布张数，达到后自动退出 (默认: 一直发布)")
        print("\n示例:")
        print("  python3 image_publisher.py test.jpg")
        print("  python3 image_publisher.py test.jpg /camera/image_raw 10")
        return
    
    image_path = sys.argv[1]
    topic_name = sys.argv[2] if len(sys.argv) > 2 else '/camera/image_raw'
    rate = float(sys.argv[3]) if len(sys.argv) > 3 else 1
    max_count = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    rclpy.init(args=args)
    node = ImagePublisher(image_path, topic_name, rate, max_count)
    if not node.ok:
        node.destroy_node()
        rclpy.shutdown()
        return
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
