#!/usr/bin/env python3
"""
将 YOLOv8 .pt 模型导出为 ONNX 格式（去掉 NMS 后处理）。
输出: <model_stem>.onnx，原始检测头输出，shape: [1, num_classes+4, num_anchors]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX without NMS")
    parser.add_argument(
        "--model",
        default="/home/juziwei/cone_ws/runs/cone_yolov8n_clean_fixed3_gpu/weights/best.pt",
        help="Path to .pt model",
    )
    parser.add_argument("--imgsz", type=int, default=960, help="Input image size")
    parser.add_argument("--output", default="", help="Output .onnx path (default: same dir as .pt)")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="Run onnx-simplifier after export")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_path = Path(args.output) if args.output else model_path.with_suffix(".onnx")

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    print(f"Classes: {model.names}")
    print(f"Exporting to ONNX (imgsz={args.imgsz}, opset={args.opset}, nms=False) ...")

    # ultralytics export: nms=False 去掉后处理，只保留原始检测头输出
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        nms=False,          # 不包含 NMS，方便 TensorRT/量化工具处理
        dynamic=False,      # 固定 batch=1，ARM 部署更友好
        half=False,         # FP32 导出，量化由下游工具完成
    )

    # ultralytics 默认把 onnx 放在 .pt 同目录，重命名到目标路径
    default_out = model_path.with_suffix(".onnx")
    if default_out.exists() and default_out != output_path:
        default_out.rename(output_path)
        print(f"Moved to: {output_path}")
    else:
        print(f"Saved to: {exported or output_path}")

    # 打印模型 IO 信息
    try:
        import onnx
        m = onnx.load(str(output_path))
        print("\n--- ONNX Model Info ---")
        for inp in m.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  Input : {inp.name}  shape={shape}")
        for out in m.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  Output: {out.name}  shape={shape}")
        print(f"  Opset : {m.opset_import[0].version}")
    except ImportError:
        print("(install onnx to see model IO info)")

    print("\nDone.")


if __name__ == "__main__":
    main()
