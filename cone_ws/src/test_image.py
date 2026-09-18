#!/usr/bin/env python3
"""
简单的图片测试脚本
用于测试 YOLO 锥桶检测，无需 ROS2
"""

import cv2
import numpy as np
from pathlib import Path

# 尝试导入 YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Run: pip install ultralytics opencv-python")
    exit(1)


def test_image(image_path, model_path, conf_threshold=0.5):
    """测试单张图片"""
    
    # 检查文件是否存在
    if not Path(image_path).exists():
        print(f"❌ 图片不存在: {image_path}")
        return False
    
    if not Path(model_path).exists():
        print(f"❌ 模型不存在: {model_path}")
        return False
    
    print(f"📷 读取图片: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ 无法读取图片")
        return False
    
    h, w = image.shape[:2]
    print(f"✓ 图片大小: {w}x{h}")
    
    # 加载模型
    print(f"🤖 加载模型: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ 无法加载模型: {e}")
        return False
    
    print(f"✓ 模型加载成功")
    
    # 运行检测
    print(f"🔍 运行检测 (conf_threshold={conf_threshold})...")
    results = model(image, conf=conf_threshold, verbose=False)
    
    # 处理结果
    debug_image = image.copy()
    detected_count = 0
    
    print(f"\n📊 检测结果:")
    print("-" * 50)
    
    if len(results[0].boxes) == 0:
        print("未检测到锥桶")
    else:
        for i, box in enumerate(results[0].boxes):
            bbox = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            x1, y1, x2, y2 = bbox.astype(int)
            
            # 计算中心点（归一化坐标）
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            
            print(f"{i+1}. 类别: {cls_name:20s} 置信度: {conf:.3f}")
            print(f"   BBox: ({x1}, {y1}) -> ({x2}, {y2})")
            print(f"   归一化坐标: ({cx:.3f}, {cy:.3f})")
            
            # 画框
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_image, f'{cls_name} {conf:.2f}', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detected_count += 1
        
        print("-" * 50)
        print(f"✓ 共检测到 {detected_count} 个锥桶\n")
    
    # 保存结果图片
    output_path = str(Path(image_path).parent / f"result_{Path(image_path).stem}.jpg")
    cv2.imwrite(output_path, debug_image)
    print(f"💾 结果已保存: {output_path}")
    
    # 显示图片（如果是 GUI 环境）
    try:
        cv2.imshow('Detection Result', debug_image)
        print("按任意键关闭图片...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass
    
    return True


def main():
    import sys
    
    # 默认参数
    model_path = '/home/juziwei/Cone Dataset/runs/cone_yolov8n/weights/best.pt'
    conf_threshold = 0.5
    
    # 获取图片路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("用法: python3 test_image.py <image_path> [model_path] [conf_threshold]")
        print(f"     model_path 默认: {model_path}")
        print(f"     conf_threshold 默认: {conf_threshold}")
        print("\n示例:")
        print("  python3 test_image.py test.jpg")
        print("  python3 test_image.py test.jpg /path/to/model.pt 0.6")
        return
    
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    
    if len(sys.argv) > 3:
        conf_threshold = float(sys.argv[3])
    
    test_image(image_path, model_path, conf_threshold)


if __name__ == '__main__':
    main()
