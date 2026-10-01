#!/usr/bin/env python3
"""Benchmark a serialized TensorRT engine on the target Jetson.

The script reports two different latency measurements:

* ``compute``: TensorRT execution with inputs and outputs already on the GPU.
* ``pipeline``: host-to-device copies, TensorRT execution, and device-to-host
  copies in one CUDA stream.

This deliberately does not include image decoding, letterbox preprocessing,
NMS, ROS transport, or message construction.  Those are measured in the ROS
detector node.  Run this script on the same Orin that runs the detector because
TensorRT engines and their performance are device/runtime specific.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as error:  # Keep --help usable on a development machine.
    np = None
    NUMPY_IMPORT_ERROR = error
else:
    NUMPY_IMPORT_ERROR = None

try:
    import tensorrt as trt
except ImportError as error:  # Keep --help usable away from the Orin.
    trt = None
    TRT_IMPORT_ERROR = error
else:
    TRT_IMPORT_ERROR = None

try:
    import torch
except ImportError as error:  # Keep --help usable away from the Orin.
    torch = None
    TORCH_IMPORT_ERROR = error
else:
    TORCH_IMPORT_ERROR = None


@dataclass
class TensorBuffer:
    name: str
    index: int
    is_input: bool
    shape: Tuple[int, ...]
    torch_dtype: object
    host: object
    device: object

    @property
    def nbytes(self) -> int:
        return int(self.host.numel() * self.host.element_size())


@dataclass
class Sample:
    phase: str
    iteration: int
    gpu_ms: float
    host_ms: float
    enqueue_ms: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure single-frame TensorRT latency on Jetson. The generated "
            "JSON contains p50/p90/p95/p99 statistics and engine metadata."
        )
    )
    parser.add_argument("--engine", type=Path, required=True, help="TensorRT .engine file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Result directory (default: <engine_dir>/benchmarks/<engine>_<timestamp>)",
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warm-up iterations (default: 50)")
    parser.add_argument(
        "--iterations", type=int, default=500, help="Measured iterations per phase (default: 500)"
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument(
        "--profile-index",
        type=int,
        default=0,
        help="TensorRT optimization profile index for dynamic engines (default: 0)",
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        metavar="NAME=1x3x640x640",
        help="Runtime shape for a dynamic input; may be repeated",
    )
    parser.add_argument(
        "--input-shape",
        metavar="1x3x640x640",
        help="Convenience form for an engine with exactly one input",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed used to initialize input tensors (default: 0)"
    )
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="Skip the H2D + engine + D2H pipeline measurement",
    )
    parser.add_argument(
        "--raw-samples",
        action="store_true",
        help="Also store every latency sample in benchmark.json (CSV is always written)",
    )
    args = parser.parse_args(argv)

    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.profile_index < 0:
        parser.error("--profile-index must be >= 0")
    return args


def parse_dims(value: str) -> Tuple[int, ...]:
    parts = [part for part in re.split(r"[xX,]", value.strip()) if part]
    if not parts:
        raise ValueError(f"Empty shape: {value!r}")
    dims = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"All runtime dimensions must be positive: {value!r}")
    return dims


def parse_shape_overrides(entries: Iterable[str]) -> Dict[str, Tuple[int, ...]]:
    shapes: Dict[str, Tuple[int, ...]] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected NAME=1x3x640x640, received: {entry!r}")
        name, raw_shape = entry.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Missing input name in: {entry!r}")
        if name in shapes:
            raise ValueError(f"Duplicate shape for input {name!r}")
        shapes[name] = parse_dims(raw_shape)
    return shapes


def percentile(values: Sequence[float], ratio: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), ratio))


def summarize(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std()),
        "min_ms": float(array.min()),
        "p50_ms": percentile(array, 50),
        "p90_ms": percentile(array, 90),
        "p95_ms": percentile(array, 95),
        "p99_ms": percentile(array, 99),
        "max_ms": float(array.max()),
        "single_stream_fps_from_mean": float(1000.0 / array.mean()),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_new_tensor_api(engine) -> bool:
    return (
        hasattr(engine, "num_io_tensors")
        and hasattr(engine, "get_tensor_name")
        and hasattr(engine, "get_tensor_mode")
    )


def tensor_records(engine) -> List[Tuple[str, int, bool]]:
    if is_new_tensor_api(engine):
        records = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            records.append((name, index, is_input))
        return records

    return [
        (engine.get_binding_name(index), index, bool(engine.binding_is_input(index)))
        for index in range(engine.num_bindings)
    ]


def engine_tensor_shape(engine, name: str, index: int) -> Tuple[int, ...]:
    if is_new_tensor_api(engine):
        return tuple(int(dim) for dim in engine.get_tensor_shape(name))
    return tuple(int(dim) for dim in engine.get_binding_shape(index))


def context_tensor_shape(context, engine, name: str, index: int) -> Tuple[int, ...]:
    if is_new_tensor_api(engine):
        return tuple(int(dim) for dim in context.get_tensor_shape(name))
    return tuple(int(dim) for dim in context.get_binding_shape(index))


def engine_tensor_dtype(engine, name: str, index: int):
    if is_new_tensor_api(engine):
        return engine.get_tensor_dtype(name)
    return engine.get_binding_dtype(index)


def profile_opt_shape(engine, name: str, index: int, profile_index: int) -> Tuple[int, ...]:
    try:
        if is_new_tensor_api(engine):
            _minimum, optimum, _maximum = engine.get_tensor_profile_shape(name, profile_index)
        else:
            _minimum, optimum, _maximum = engine.get_profile_shape(profile_index, index)
    except Exception as error:
        raise RuntimeError(
            f"Input {name!r} is dynamic. Supply --shape {name}=... because its "
            f"profile shape could not be read: {error}"
        ) from error
    return tuple(int(dim) for dim in optimum)


def set_input_shape(context, engine, name: str, index: int, shape: Tuple[int, ...]) -> None:
    if is_new_tensor_api(engine):
        accepted = context.set_input_shape(name, shape)
    else:
        accepted = context.set_binding_shape(index, shape)
    if accepted is False:
        raise RuntimeError(f"TensorRT rejected runtime shape {shape} for input {name!r}")


def resolve_input_shapes(
    engine,
    context,
    records: Sequence[Tuple[str, int, bool]],
    shape_overrides: Dict[str, Tuple[int, ...]],
    input_shape: Optional[str],
    profile_index: int,
) -> Dict[str, Tuple[int, ...]]:
    input_records = [record for record in records if record[2]]
    if input_shape:
        if len(input_records) != 1:
            raise ValueError("--input-shape requires an engine with exactly one input")
        only_name = input_records[0][0]
        if only_name in shape_overrides:
            raise ValueError(f"Use either --input-shape or --shape {only_name}=..., not both")
        shape_overrides[only_name] = parse_dims(input_shape)

    valid_names = {name for name, _index, _is_input in input_records}
    unknown_names = sorted(set(shape_overrides) - valid_names)
    if unknown_names:
        raise ValueError(
            f"Unknown input name(s): {unknown_names}. Engine inputs: {sorted(valid_names)}"
        )

    resolved: Dict[str, Tuple[int, ...]] = {}
    for name, index, _is_input in input_records:
        declared = engine_tensor_shape(engine, name, index)
        dynamic = any(dim < 0 for dim in declared)
        requested = shape_overrides.get(name)

        if dynamic:
            shape = requested or profile_opt_shape(engine, name, index, profile_index)
            set_input_shape(context, engine, name, index, shape)
        else:
            shape = declared
            if requested and requested != declared:
                raise ValueError(
                    f"Input {name!r} has fixed shape {declared}; requested {requested}. "
                    "Rebuild the engine for the new resolution."
                )
        resolved[name] = shape

    return resolved


def torch_dtype_for_trt(dtype):
    """Map TensorRT data types without going through NumPy's binary API."""
    candidates = [
        ("float32", "float32"),
        ("float16", "float16"),
        ("int8", "int8"),
        ("int32", "int32"),
        ("bool", "bool"),
        ("uint8", "uint8"),
        ("int64", "int64"),
        ("bfloat16", "bfloat16"),
    ]
    for trt_name, torch_name in candidates:
        if hasattr(trt, trt_name) and dtype == getattr(trt, trt_name):
            return getattr(torch, torch_name)

    # TensorRT releases differ in which aliases are exported at module level.
    enum_candidates = [
        ("FLOAT", "float32"),
        ("HALF", "float16"),
        ("INT8", "int8"),
        ("INT32", "int32"),
        ("BOOL", "bool"),
        ("UINT8", "uint8"),
        ("INT64", "int64"),
        ("BF16", "bfloat16"),
    ]
    data_type = getattr(trt, "DataType", None)
    if data_type is not None:
        for enum_name, torch_name in enum_candidates:
            if hasattr(data_type, enum_name) and dtype == getattr(data_type, enum_name):
                return getattr(torch, torch_name)
    raise TypeError(f"Unsupported TensorRT tensor dtype: {dtype}")


def fill_input(host, dtype, generator) -> None:
    if dtype.is_floating_point:
        host.uniform_(0.0, 1.0, generator=generator)
    elif dtype == torch.bool:
        temporary = torch.randint(
            0, 2, host.shape, dtype=torch.uint8, generator=generator
        )
        host.copy_(temporary)
    else:
        info = torch.iinfo(dtype)
        low = max(info.min, -16)
        high = min(info.max, 16)
        host.random_(low, high + 1, generator=generator)


def allocate_buffers(
    engine,
    context,
    records: Sequence[Tuple[str, int, bool]],
    generator,
    device,
) -> Tuple[List[TensorBuffer], List[int]]:
    buffers: List[TensorBuffer] = []
    bindings = [0] * (engine.num_bindings if not is_new_tensor_api(engine) else len(records))

    for name, index, is_input in records:
        shape = context_tensor_shape(context, engine, name, index)
        if not shape or any(dim < 0 for dim in shape):
            raise RuntimeError(
                f"Tensor {name!r} has unresolved/data-dependent shape {shape}. "
                "This benchmark currently requires concrete input and output shapes."
            )
        volume = int(trt.volume(shape))
        if volume <= 0:
            raise RuntimeError(f"Tensor {name!r} has invalid shape {shape}")

        torch_dtype = torch_dtype_for_trt(engine_tensor_dtype(engine, name, index))
        host = torch.empty(volume, dtype=torch_dtype, pin_memory=True)
        if is_input:
            fill_input(host, torch_dtype, generator)
        else:
            host.zero_()
        device_tensor = torch.empty(volume, dtype=torch_dtype, device=device)
        buffer = TensorBuffer(
            name, index, is_input, shape, torch_dtype, host, device_tensor
        )
        buffers.append(buffer)

        if is_new_tensor_api(engine):
            context.set_tensor_address(name, int(device_tensor.data_ptr()))
        else:
            bindings[index] = int(device_tensor.data_ptr())

    return buffers, bindings


def execute_async(context, engine, bindings: Sequence[int], stream) -> None:
    stream_handle = int(stream.cuda_stream)
    if is_new_tensor_api(engine) and hasattr(context, "execute_async_v3"):
        ok = context.execute_async_v3(stream_handle=stream_handle)
    else:
        ok = context.execute_async_v2(bindings=bindings, stream_handle=stream_handle)
    if ok is False:
        raise RuntimeError("TensorRT execute_async returned false")


def copy_inputs_to_device(buffers: Sequence[TensorBuffer], stream) -> None:
    with torch.cuda.stream(stream):
        for buffer in buffers:
            if buffer.is_input:
                buffer.device.copy_(buffer.host, non_blocking=True)


def copy_outputs_to_host(buffers: Sequence[TensorBuffer], stream) -> None:
    with torch.cuda.stream(stream):
        for buffer in buffers:
            if not buffer.is_input:
                buffer.host.copy_(buffer.device, non_blocking=True)


def warm_up(
    context,
    engine,
    bindings: Sequence[int],
    buffers: Sequence[TensorBuffer],
    stream,
    iterations: int,
) -> None:
    copy_inputs_to_device(buffers, stream)
    for _ in range(iterations):
        execute_async(context, engine, bindings, stream)
    stream.synchronize()


def measure_phase(
    phase: str,
    context,
    engine,
    bindings: Sequence[int],
    buffers: Sequence[TensorBuffer],
    stream,
    iterations: int,
    include_transfers: bool,
) -> List[Sample]:
    samples: List[Sample] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if not include_transfers:
        copy_inputs_to_device(buffers, stream)
        stream.synchronize()

    for iteration in range(iterations):
        host_start = time.perf_counter_ns()
        start_event.record(stream)
        if include_transfers:
            copy_inputs_to_device(buffers, stream)

        enqueue_start = time.perf_counter_ns()
        execute_async(context, engine, bindings, stream)
        enqueue_end = time.perf_counter_ns()

        if include_transfers:
            copy_outputs_to_host(buffers, stream)
        end_event.record(stream)
        end_event.synchronize()
        host_end = time.perf_counter_ns()

        samples.append(
            Sample(
                phase=phase,
                iteration=iteration,
                gpu_ms=float(start_event.elapsed_time(end_event)),
                host_ms=(host_end - host_start) * 1e-6,
                enqueue_ms=(enqueue_end - enqueue_start) * 1e-6,
            )
        )
    return samples


def phase_summary(samples: Sequence[Sample]) -> Dict[str, Dict[str, float]]:
    return {
        "gpu": summarize([sample.gpu_ms for sample in samples]),
        "host": summarize([sample.host_ms for sample in samples]),
        "enqueue": summarize([sample.enqueue_ms for sample in samples]),
    }


def write_samples(path: Path, samples: Sequence[Sample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["phase", "iteration", "gpu_ms", "host_ms", "enqueue_ms"]
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "phase": sample.phase,
                    "iteration": sample.iteration,
                    "gpu_ms": f"{sample.gpu_ms:.6f}",
                    "host_ms": f"{sample.host_ms:.6f}",
                    "enqueue_ms": f"{sample.enqueue_ms:.6f}",
                }
            )


def print_summary(label: str, summary: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{label}")
    print("  metric          mean      p50      p90      p95      p99      max")
    for metric in ("gpu", "host", "enqueue"):
        values = summary[metric]
        print(
            f"  {metric:<8} "
            f"{values['mean_ms']:9.3f} {values['p50_ms']:8.3f} "
            f"{values['p90_ms']:8.3f} {values['p95_ms']:8.3f} "
            f"{values['p99_ms']:8.3f} {values['max_ms']:8.3f} ms"
        )


def export_engine_info(engine, output_path: Path) -> Optional[str]:
    if not hasattr(engine, "create_engine_inspector"):
        return "TensorRT runtime does not provide EngineInspector"
    try:
        inspector = engine.create_engine_inspector()
        information = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        output_path.write_text(information, encoding="utf-8")
        return None
    except Exception as error:
        return str(error)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if np is None:
        print(f"NumPy is unavailable: {NUMPY_IMPORT_ERROR}", file=sys.stderr)
        return 2
    if trt is None:
        print(f"TensorRT Python module is unavailable: {TRT_IMPORT_ERROR}", file=sys.stderr)
        return 2
    if torch is None:
        print(f"PyTorch is unavailable: {TORCH_IMPORT_ERROR}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("PyTorch cannot access CUDA on this system", file=sys.stderr)
        return 2

    engine_path = args.engine.expanduser().resolve()
    if not engine_path.is_file():
        print(f"Engine not found: {engine_path}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else engine_path.parent / "benchmarks" / f"{engine_path.stem}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        shape_overrides = parse_shape_overrides(args.shape)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    torch.cuda.set_device(args.device)
    torch.cuda.init()
    device = torch.device("cuda", args.device)
    device_properties = torch.cuda.get_device_properties(args.device)

    try:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine_bytes = engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(
                "TensorRT could not deserialize the engine. Check TensorRT/CUDA version "
                "and whether the engine was built for this Orin."
            )
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT could not create an execution context")

        profile_count = int(getattr(engine, "num_optimization_profiles", 1))
        if args.profile_index >= profile_count:
            raise ValueError(
                f"Profile index {args.profile_index} is out of range; engine has {profile_count} profile(s)"
            )
        if profile_count > 1 and hasattr(context, "active_optimization_profile"):
            context.active_optimization_profile = args.profile_index

        records = tensor_records(engine)
        input_shapes = resolve_input_shapes(
            engine,
            context,
            records,
            shape_overrides,
            args.input_shape,
            args.profile_index,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)
        buffers, bindings = allocate_buffers(
            engine, context, records, generator, device
        )
        stream = torch.cuda.Stream(device=device)

        print(f"Engine: {engine_path}")
        print(f"TensorRT: {trt.__version__}")
        print(
            f"CUDA device: {device_properties.name} | compute capability "
            f"{device_properties.major}.{device_properties.minor}"
        )
        print("Tensors:")
        for buffer in buffers:
            direction = "input " if buffer.is_input else "output"
            print(
                f"  {direction} {buffer.name}: shape={buffer.shape} "
                f"dtype={buffer.torch_dtype} bytes={buffer.nbytes}"
            )

        inspector_error = export_engine_info(engine, output_dir / "engine_info.json")

        print(f"\nWarm-up: {args.warmup} iterations")
        warm_up(context, engine, bindings, buffers, stream, args.warmup)

        print(f"Measuring compute-only latency: {args.iterations} iterations")
        compute_samples = measure_phase(
            "compute",
            context,
            engine,
            bindings,
            buffers,
            stream,
            args.iterations,
            include_transfers=False,
        )
        all_samples = list(compute_samples)
        summaries = {"compute": phase_summary(compute_samples)}
        print_summary("Compute only (device buffers already prepared)", summaries["compute"])

        if not args.compute_only:
            print(f"\nMeasuring transfer + compute latency: {args.iterations} iterations")
            pipeline_samples = measure_phase(
                "pipeline",
                context,
                engine,
                bindings,
                buffers,
                stream,
                args.iterations,
                include_transfers=True,
            )
            all_samples.extend(pipeline_samples)
            summaries["pipeline"] = phase_summary(pipeline_samples)
            print_summary("H2D + TensorRT + D2H", summaries["pipeline"])

        metadata = {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "command": sys.argv,
            "engine": {
                "path": str(engine_path),
                "size_bytes": engine_path.stat().st_size,
                "sha256": sha256_file(engine_path),
                "profile_index": args.profile_index,
            },
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "tensorrt": trt.__version__,
                "pytorch": torch.__version__,
                "pytorch_cuda": torch.version.cuda,
                "cuda_device_index": args.device,
                "cuda_device_name": device_properties.name,
                "compute_capability": [
                    device_properties.major,
                    device_properties.minor,
                ],
                "device_total_memory_bytes": int(device_properties.total_memory),
            },
            "settings": {
                "warmup": args.warmup,
                "iterations": args.iterations,
                "seed": args.seed,
                "compute_only": args.compute_only,
            },
            "input_shapes": {name: list(shape) for name, shape in input_shapes.items()},
            "tensors": [
                {
                    "name": buffer.name,
                    "direction": "input" if buffer.is_input else "output",
                    "shape": list(buffer.shape),
                    "dtype": str(buffer.torch_dtype),
                    "bytes": buffer.nbytes,
                }
                for buffer in buffers
            ],
            "summary": summaries,
            "engine_inspector_error": inspector_error,
        }
        if args.raw_samples:
            metadata["samples"] = [sample.__dict__ for sample in all_samples]

        write_samples(output_dir / "latencies.csv", all_samples)
        (output_dir / "benchmark.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nResults: {output_dir}")
        print("  benchmark.json  summary and reproducibility metadata")
        print("  latencies.csv   every measured iteration")
        if inspector_error:
            print(f"  engine inspector unavailable: {inspector_error}")
        else:
            print("  engine_info.json TensorRT engine inspector output")
        return 0
    except Exception as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
