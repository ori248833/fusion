#!/usr/bin/env python3
"""Export a YOLO PT model to ONNX and build a calibrated TensorRT INT8 engine.

All user settings live in the configuration section below.  The script is
intended to run on the deployment Jetson so the generated engine matches its
TensorRT, CUDA and GPU versions.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple


# =============================================================================
# 用户配置区：修改这一段，然后运行 python3 scripts/build_int8_engine.py
# =============================================================================

# 输入PT模型以及中间ONNX、最终engine的保存位置。
PT_MODEL_PATH = 'models/best.pt'
ONNX_OUTPUT_PATH = 'models/best_int8_640.onnx'
ENGINE_OUTPUT_PATH = 'models/best_int8_640.engine'

# 校准图片目录。可以递归包含jpg/png等图片，不需要标签。
# 建议使用300～1000张具有代表性的真实相机图像，覆盖远近锥桶、光照和赛道场景。
CALIBRATION_IMAGE_DIR = '/path/to/calibration/images'
CALIBRATION_CACHE_PATH = 'runs/tensorrt_build/best_int8_640.cache'

# 构建过程与实际层格式报告。
BUILD_REPORT_PATH = 'runs/tensorrt_build/best_int8_640_report.json'
LAYER_INSPECTOR_PATH = 'runs/tensorrt_build/best_int8_640_layers.json'

# 固定输入规格。部署时yolo_detector.yaml中的imgsz必须与这里一致。
INPUT_HEIGHT = 640
INPUT_WIDTH = 640
BATCH_SIZE = 1

# ONNX导出设置。REEXPORT_ONNX=False时会复用已有ONNX。
REEXPORT_ONNX = True
ONNX_OPSET = 11
SIMPLIFY_ONNX = False

# INT8校准预处理必须与实际部署预处理一致。
# 当前YOLO节点没有裁剪ROI，因此默认必须为0.0。
CROP_TOP_RATIO = 0.0
LETTERBOX_COLOR = (114, 114, 114)
MAX_CALIBRATION_IMAGES = 500  # 0表示使用目录中的全部图片。
REUSE_CALIBRATION_CACHE = True

# 分层精度策略：
# aggressive   检测头cv2/cv3卷积分支保留INT8，stem与DFL/解码部分回退FP16。
# conservative 整个检测头回退FP16，通常精度更稳但速度可能略慢。
# full_int8    不主动设置FP16层，让TensorRT尽量自行选择INT8。
# custom       只使用下面的CUSTOM_FP16/FP32_PATTERNS。
PRECISION_PROFILE = 'aggressive'

# 名称匹配不区分大小写，并且使用“包含”匹配。FP32规则优先于FP16规则。
CUSTOM_FP16_PATTERNS: Sequence[str] = ()
CUSTOM_FP32_PATTERNS: Sequence[str] = ()

# TensorRT构建资源与策略搜索设置。
WORKSPACE_GIB = 4
BUILDER_OPTIMIZATION_LEVEL = 5
AVERAGE_TIMING_ITERATIONS = 8
MAX_AUX_STREAMS = 0
ENABLE_SPARSE_WEIGHTS = False
DISABLE_TF32 = False

# =============================================================================
# 用户配置区结束：通常不需要修改下面的代码
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_signature(paths: Sequence[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
        record = f'{relative}|{stat.st_size}|{stat.st_mtime_ns}\n'
        digest.update(record.encode('utf-8', errors='replace'))
    return digest.hexdigest()


def collect_calibration_images(directory: Path) -> List[Path]:
    paths = sorted(
        path
        for path in directory.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if MAX_CALIBRATION_IMAGES > 0:
        paths = paths[:MAX_CALIBRATION_IMAGES]
    if not paths:
        raise RuntimeError(f'No calibration images found in: {directory}')
    return paths


def validate_configuration() -> Dict[str, Path]:
    paths = {
        'pt': resolve_path(PT_MODEL_PATH),
        'onnx': resolve_path(ONNX_OUTPUT_PATH),
        'engine': resolve_path(ENGINE_OUTPUT_PATH),
        'calibration_dir': resolve_path(CALIBRATION_IMAGE_DIR),
        'cache': resolve_path(CALIBRATION_CACHE_PATH),
        'report': resolve_path(BUILD_REPORT_PATH),
        'inspector': resolve_path(LAYER_INSPECTOR_PATH),
    }
    if not paths['pt'].is_file():
        raise FileNotFoundError(f'PT model not found: {paths["pt"]}')
    if not paths['calibration_dir'].is_dir():
        raise FileNotFoundError(
            f'Calibration image directory not found: {paths["calibration_dir"]}'
        )
    if paths['onnx'].suffix.lower() != '.onnx':
        raise ValueError('ONNX_OUTPUT_PATH must end in .onnx')
    if paths['engine'].suffix.lower() != '.engine':
        raise ValueError('ENGINE_OUTPUT_PATH must end in .engine')
    if paths['report'].suffix.lower() != '.json':
        raise ValueError('BUILD_REPORT_PATH must end in .json')
    if INPUT_HEIGHT <= 0 or INPUT_WIDTH <= 0:
        raise ValueError('INPUT_HEIGHT and INPUT_WIDTH must be positive')
    if INPUT_HEIGHT % 32 or INPUT_WIDTH % 32:
        raise ValueError('INPUT_HEIGHT and INPUT_WIDTH must be multiples of 32')
    if BATCH_SIZE != 1:
        raise ValueError('This deployment script currently requires BATCH_SIZE = 1')
    if not 0.0 <= CROP_TOP_RATIO < 1.0:
        raise ValueError('CROP_TOP_RATIO must be in [0, 1)')
    if PRECISION_PROFILE not in {'aggressive', 'conservative', 'full_int8', 'custom'}:
        raise ValueError(f'Unknown PRECISION_PROFILE: {PRECISION_PROFILE}')
    if WORKSPACE_GIB <= 0:
        raise ValueError('WORKSPACE_GIB must be positive')
    return paths


def import_dependencies():
    try:
        import cv2
        import numpy as np
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            'Missing build dependency. Run this script in the Orin environment '
            f'containing Ultralytics, TensorRT, PyCUDA, OpenCV and NumPy: {error}'
        ) from error
    return cv2, np, cuda, trt, YOLO


def export_onnx(pt_path: Path, onnx_path: Path, yolo_class) -> Path:
    if onnx_path.is_file() and not REEXPORT_ONNX:
        print(f'Reusing ONNX: {onnx_path}')
        return onnx_path

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Exporting PT to ONNX: {pt_path}')
    model = yolo_class(str(pt_path))
    exported = model.export(
        format='onnx',
        imgsz=(INPUT_HEIGHT, INPUT_WIDTH),
        batch=BATCH_SIZE,
        opset=ONNX_OPSET,
        simplify=SIMPLIFY_ONNX,
        nms=False,
        dynamic=False,
        half=False,
    )
    exported_path = Path(str(exported)).expanduser().resolve() if exported else None
    if exported_path is None or not exported_path.is_file():
        fallback = pt_path.with_suffix('.onnx')
        exported_path = fallback if fallback.is_file() else None
    if exported_path is None:
        raise RuntimeError('Ultralytics did not return a usable ONNX path')
    if exported_path != onnx_path:
        shutil.copy2(exported_path, onnx_path)
    print(f'ONNX ready: {onnx_path}')
    return onnx_path


def make_calibrator(
    trt,
    cuda,
    cv2,
    np,
    image_paths: Sequence[Path],
    cache_path: Path,
    expected_cache_metadata: Dict[str, object],
):
    metadata_path = cache_path.with_suffix(cache_path.suffix + '.json')

    class YOLOMinMaxCalibrator(trt.IInt8MinMaxCalibrator):
        def __init__(self):
            super().__init__()
            self.image_paths = list(image_paths)
            self.target_h = INPUT_HEIGHT
            self.target_w = INPUT_WIDTH
            self.batch_size = BATCH_SIZE
            self.host_batch = np.empty(
                (self.batch_size, 3, self.target_h, self.target_w),
                dtype=np.float32,
            )
            self.device_input = cuda.mem_alloc(self.host_batch.nbytes)
            self.generator = self._batch_generator()
            self.valid_images = 0
            self.skipped_images = 0

        def _preprocess(self, image):
            if CROP_TOP_RATIO > 0.0:
                crop_start = int(image.shape[0] * CROP_TOP_RATIO)
                image = image[crop_start:, :]
            source_h, source_w = image.shape[:2]
            scale = min(self.target_w / source_w, self.target_h / source_h)
            resized_w = max(1, round(source_w * scale))
            resized_h = max(1, round(source_h * scale))
            resized = cv2.resize(
                image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
            )
            left = (self.target_w - resized_w) // 2
            right = self.target_w - resized_w - left
            top = (self.target_h - resized_h) // 2
            bottom = self.target_h - resized_h - top
            padded = cv2.copyMakeBorder(
                resized,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=LETTERBOX_COLOR,
            )
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            return np.ascontiguousarray(chw)

        def _batch_generator(self):
            pending = []
            for image_path in self.image_paths:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None or image.size == 0:
                    self.skipped_images += 1
                    print(f'Warning: unreadable calibration image: {image_path}')
                    continue
                pending.append(self._preprocess(image))
                self.valid_images += 1
                if len(pending) == self.batch_size:
                    yield np.stack(pending, axis=0)
                    pending.clear()
            if pending:
                while len(pending) < self.batch_size:
                    pending.append(pending[-1])
                yield np.stack(pending, axis=0)

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            del names
            try:
                batch = next(self.generator)
            except StopIteration:
                return None
            np.copyto(self.host_batch, batch)
            cuda.memcpy_htod(self.device_input, self.host_batch)
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if not REUSE_CALIBRATION_CACHE or not cache_path.is_file():
                return None
            if not metadata_path.is_file():
                print('Calibration cache metadata is missing; recalibrating.')
                return None
            try:
                saved = json.loads(metadata_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                print('Calibration cache metadata is invalid; recalibrating.')
                return None
            if saved != expected_cache_metadata:
                print('Calibration settings changed; ignoring the old cache.')
                return None
            print(f'Reusing calibration cache: {cache_path}')
            return cache_path.read_bytes()

        def write_calibration_cache(self, cache):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(cache)
            metadata_path.write_text(
                json.dumps(expected_cache_metadata, indent=2, ensure_ascii=False),
                encoding='utf-8',
            )

    return YOLOMinMaxCalibrator()


def name_contains(name: str, patterns: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(pattern.lower() in lowered for pattern in patterns if pattern)


def profile_precision(layer_name: str) -> Optional[str]:
    lowered = layer_name.lower()
    if name_contains(layer_name, CUSTOM_FP32_PATTERNS):
        return 'fp32'
    if name_contains(layer_name, CUSTOM_FP16_PATTERNS):
        return 'fp16'
    if PRECISION_PROFILE in {'full_int8', 'custom'}:
        return None
    in_head = 'model.22' in lowered
    in_stem = 'model.0' in lowered
    if PRECISION_PROFILE == 'conservative' and in_head:
        return 'fp16'
    if PRECISION_PROFILE == 'aggressive':
        safe_head_convolution = 'cv2' in lowered or 'cv3' in lowered
        if in_stem or (in_head and not safe_head_convolution):
            return 'fp16'
    return None


def apply_precision_policy(network, trt) -> Dict[str, object]:
    unsafe_types = {
        getattr(trt.LayerType, name)
        for name in ('CONSTANT', 'SHAPE', 'GATHER', 'CAST', 'CONCATENATION')
        if hasattr(trt.LayerType, name)
    }
    non_float_types = {
        data_type
        for data_type in (
            getattr(trt.DataType, 'INT32', None),
            getattr(trt.DataType, 'INT64', None),
            getattr(trt.DataType, 'BOOL', None),
        )
        if data_type is not None
    }
    requested = []
    skipped = []
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        target = profile_precision(layer.name)
        if target is None:
            continue
        if layer.type in unsafe_types:
            skipped.append({'name': layer.name, 'reason': f'unsafe type {layer.type}'})
            continue
        outputs: List[Tuple[int, object]] = []
        unsafe_output = False
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if tensor is None:
                continue
            if getattr(tensor, 'is_shape_tensor', False) or tensor.dtype in non_float_types:
                unsafe_output = True
                break
            outputs.append((output_index, tensor))
        if unsafe_output:
            skipped.append({'name': layer.name, 'reason': 'shape/integer output'})
            continue
        data_type = trt.DataType.FLOAT if target == 'fp32' else trt.DataType.HALF
        try:
            layer.precision = data_type
            for output_index, _ in outputs:
                layer.set_output_type(output_index, data_type)
        except (AttributeError, RuntimeError) as error:
            skipped.append({'name': layer.name, 'reason': str(error)})
            continue
        requested.append({'name': layer.name, 'precision': target})
    return {
        'profile': PRECISION_PROFILE,
        'requested_counts': dict(Counter(item['precision'] for item in requested)),
        'requested_layers': requested,
        'skipped_layers': skipped,
    }


def parse_network(parser, onnx_path: Path) -> None:
    if parser.parse(onnx_path.read_bytes()):
        return
    errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
    raise RuntimeError('ONNX parse failed:\n' + '\n'.join(errors))


def set_builder_options(config, trt) -> None:
    workspace_bytes = int(WORKSPACE_GIB * (1 << 30))
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes
    if hasattr(config, 'builder_optimization_level'):
        config.builder_optimization_level = BUILDER_OPTIMIZATION_LEVEL
    if hasattr(config, 'avg_timing_iterations'):
        config.avg_timing_iterations = AVERAGE_TIMING_ITERATIONS
    if hasattr(config, 'max_aux_streams'):
        config.max_aux_streams = MAX_AUX_STREAMS
    if hasattr(config, 'profiling_verbosity') and hasattr(trt, 'ProfilingVerbosity'):
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if ENABLE_SPARSE_WEIGHTS and hasattr(trt.BuilderFlag, 'SPARSE_WEIGHTS'):
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    if DISABLE_TF32 and hasattr(trt.BuilderFlag, 'TF32'):
        config.clear_flag(trt.BuilderFlag.TF32)


def configure_precision(config, trt, calibrator) -> None:
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    if hasattr(trt.BuilderFlag, 'OBEY_PRECISION_CONSTRAINTS'):
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    elif hasattr(trt.BuilderFlag, 'STRICT_TYPES'):
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    config.int8_calibrator = calibrator


def engine_io(engine, trt) -> List[Dict[str, object]]:
    records = []
    if hasattr(engine, 'num_io_tensors'):
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            records.append(
                {
                    'name': name,
                    'mode': str(mode),
                    'shape': list(engine.get_tensor_shape(name)),
                    'dtype': str(engine.get_tensor_dtype(name)),
                }
            )
        return records
    for index in range(engine.num_bindings):
        records.append(
            {
                'name': engine.get_binding_name(index),
                'mode': 'INPUT' if engine.binding_is_input(index) else 'OUTPUT',
                'shape': list(engine.get_binding_shape(index)),
                'dtype': str(engine.get_binding_dtype(index)),
            }
        )
    return records


def inspect_engine(engine, trt, inspector_path: Path) -> Dict[str, object]:
    if not hasattr(engine, 'create_engine_inspector'):
        return {'available': False, 'reason': 'EngineInspector is unavailable'}
    inspector = engine.create_engine_inspector()
    try:
        raw = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    except (AttributeError, RuntimeError):
        raw = inspector.get_engine_information(trt.LayerInformationFormat.ONELINE)
    inspector_path.parent.mkdir(parents=True, exist_ok=True)
    inspector_path.write_text(raw, encoding='utf-8')

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            layers = parsed.get('Layers', parsed.get('layers', [parsed]))
        else:
            layers = parsed
        if not isinstance(layers, list):
            layers = [layers]
    except json.JSONDecodeError:
        layers = [line for line in raw.splitlines() if line.strip()]

    precision_groups = Counter()
    reformat_count = 0
    for layer in layers:
        description = (
            json.dumps(layer, ensure_ascii=False)
            if isinstance(layer, dict)
            else str(layer)
        )
        lowered = description.lower()
        formats = []
        if 'int8' in lowered:
            formats.append('int8')
        if 'half' in lowered or 'fp16' in lowered:
            formats.append('fp16')
        if 'float' in lowered or 'fp32' in lowered:
            formats.append('fp32')
        precision_groups['+'.join(formats) if formats else 'unknown'] += 1
        if 'reformat' in lowered or 'format conversion' in lowered:
            reformat_count += 1
    return {
        'available': True,
        'raw_report': str(inspector_path),
        'layer_records': len(layers),
        'precision_group_mentions': dict(precision_groups),
        'reformat_layer_count': reformat_count,
        'note': (
            'Precision groups are summarized from inspector text. The raw report '
            'is authoritative because TensorRT may fuse several ONNX layers.'
        ),
    }


def build_engine(paths: Dict[str, Path], image_paths: Sequence[Path], modules):
    cv2, np, cuda, trt, _ = modules
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    parse_network(parser, paths['onnx'])
    if network.num_inputs != 1:
        raise RuntimeError(f'Expected one model input, found {network.num_inputs}')
    expected_shape = (BATCH_SIZE, 3, INPUT_HEIGHT, INPUT_WIDTH)
    actual_shape = tuple(network.get_input(0).shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f'ONNX input shape {actual_shape} does not match configured {expected_shape}'
        )
    if hasattr(builder, 'platform_has_fast_int8') and not builder.platform_has_fast_int8:
        print('Warning: TensorRT reports no fast native INT8 support on this platform.')

    cache_metadata = {
        'schema_version': 1,
        'onnx_sha256': sha256_file(paths['onnx']),
        'input_shape': list(expected_shape),
        'crop_top_ratio': CROP_TOP_RATIO,
        'letterbox_color': list(LETTERBOX_COLOR),
        'preprocessing': 'letterbox_rgb_chw_float32_0_1',
        'calibration_image_count': len(image_paths),
        'calibration_signature': calibration_signature(
            image_paths, paths['calibration_dir']
        ),
    }
    calibrator = make_calibrator(
        trt,
        cuda,
        cv2,
        np,
        image_paths,
        paths['cache'],
        cache_metadata,
    )
    set_builder_options(config, trt)
    configure_precision(config, trt, calibrator)
    precision_policy = apply_precision_policy(network, trt)
    print(
        'Precision constraints requested: '
        f'{precision_policy["requested_counts"]}; '
        f'skipped={len(precision_policy["skipped_layers"])}'
    )
    print('Building TensorRT engine. This may take several minutes...')
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError('TensorRT returned an empty serialized engine')

    paths['engine'].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths['engine'].with_suffix(paths['engine'].suffix + '.tmp')
    temporary.write_bytes(bytes(serialized))
    temporary.replace(paths['engine'])

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(paths['engine'].read_bytes())
    if engine is None:
        raise RuntimeError('The new engine could not be deserialized')
    inspection = inspect_engine(engine, trt, paths['inspector'])
    return {
        'build_seconds': build_seconds,
        'engine_io': engine_io(engine, trt),
        'precision_policy': precision_policy,
        'inspection': inspection,
        'calibration': {
            **cache_metadata,
            'cache_path': str(paths['cache']),
            'valid_images_seen_during_new_calibration': calibrator.valid_images,
            'skipped_images': calibrator.skipped_images,
        },
    }


def main() -> int:
    try:
        paths = validate_configuration()
        modules = import_dependencies()
        image_paths = collect_calibration_images(paths['calibration_dir'])
        export_onnx(paths['pt'], paths['onnx'], modules[-1])
        build_result = build_engine(paths, image_paths, modules)
    except Exception as error:
        print(f'Build failed: {type(error).__name__}: {error}', file=sys.stderr)
        return 1

    _, _, _, trt, _ = modules
    report = {
        'schema_version': 1,
        'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'tensorrt_version': getattr(trt, '__version__', None),
        'configuration': {
            'pt_model': str(paths['pt']),
            'onnx_output': str(paths['onnx']),
            'engine_output': str(paths['engine']),
            'input_shape': [BATCH_SIZE, 3, INPUT_HEIGHT, INPUT_WIDTH],
            'precision_profile': PRECISION_PROFILE,
            'workspace_gib': WORKSPACE_GIB,
            'crop_top_ratio': CROP_TOP_RATIO,
            'reexport_onnx': REEXPORT_ONNX,
        },
        'artifacts': {
            'pt_sha256': sha256_file(paths['pt']),
            'onnx_sha256': sha256_file(paths['onnx']),
            'engine_sha256': sha256_file(paths['engine']),
            'engine_bytes': paths['engine'].stat().st_size,
        },
        'build': build_result,
    }
    paths['report'].parent.mkdir(parents=True, exist_ok=True)
    paths['report'].write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'Engine written to: {paths["engine"]}')
    print(f'Build report written to: {paths["report"]}')
    print(f'Layer inspector written to: {paths["inspector"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
