#!/usr/bin/env python3
"""Lightweight NVIDIA Jetson Orin utilization and GPU-process watcher.

The script uses tegrastats for board-wide metrics.  It also scans /proc to
identify processes that have opened NVIDIA GPU devices or mapped CUDA,
TensorRT, cuDNN, ZED, or graphics libraries.

Important: Jetson's normal tegrastats interface reports board-wide GPU load,
not a trustworthy per-process GPU percentage.  The process table therefore
shows GPU users/candidates plus CPU and resident memory, without inventing a
per-process GPU utilization number.  Use Nsight Systems for exact CUDA kernel
and GPU-time attribution.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


RAM_RE = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
SWAP_RE = re.compile(r"\bSWAP\s+(\d+)/(\d+)MB")
CPU_RE = re.compile(r"\bCPU\s*\[([^]]*)\]")
GPU_RE = re.compile(r"\bGR3D_FREQ\s+(\d+(?:\.\d+)?)%(?:@\[?([0-9,]+)\]?)?")
EMC_RE = re.compile(r"\bEMC_FREQ\s+(\d+(?:\.\d+)?)%(?:@(\d+))?")
TEMP_RE = re.compile(r"\b([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C")
POWER_RE = re.compile(r"\b(?:VDD_IN|POM_5V_IN)\s+(\d+)mW/(\d+)mW")
CPU_ENTRY_RE = re.compile(r"(\d+(?:\.\d+)?)%@")

GPU_LIBRARY_HINTS: Sequence[Tuple[str, str]] = (
    ("libcuda", "CUDA"),
    ("libnvinfer", "TensorRT"),
    ("libnvonnxparser", "TensorRT"),
    ("libcudnn", "cuDNN"),
    ("libcublas", "cuBLAS"),
    ("libcufft", "cuFFT"),
    ("libnvidia-egl", "EGL"),
    ("libnvidia-gl", "OpenGL"),
    ("libopencv_cud", "OpenCV-CUDA"),
    ("libsl_zed", "ZED"),
)

COMMAND_HINTS: Sequence[Tuple[str, str]] = (
    ("zed", "ZED-name"),
    ("rviz", "RViz-name"),
    ("yolo", "YOLO-name"),
    ("pointpillar", "PointPillars-name"),
    ("tensorrt", "TensorRT-name"),
    ("trtexec", "TensorRT-name"),
)


@dataclass
class BoardSample:
    timestamp: str
    elapsed_s: float
    tag: str
    gpu_pct: Optional[float]
    gpu_mhz: Optional[int]
    emc_pct: Optional[float]
    emc_mhz: Optional[int]
    cpu_avg_pct: Optional[float]
    cpu_max_pct: Optional[float]
    ram_used_mb: Optional[int]
    ram_total_mb: Optional[int]
    swap_used_mb: Optional[int]
    swap_total_mb: Optional[int]
    gpu_temp_c: Optional[float]
    cpu_temp_c: Optional[float]
    max_temp_c: Optional[float]
    input_power_mw: Optional[int]
    input_power_avg_mw: Optional[int]
    raw: str


@dataclass
class ProcessSample:
    timestamp: str
    elapsed_s: float
    tag: str
    pid: int
    cpu_pct: Optional[float]
    rss_mb: Optional[float]
    evidence: str
    command: str


class CpuTracker:
    def __init__(self) -> None:
        self._previous: Dict[int, Tuple[int, float]] = {}
        self._clock_ticks = os.sysconf("SC_CLK_TCK")

    def utilization(self, pid: int, ticks: int, now: float) -> Optional[float]:
        previous = self._previous.get(pid)
        self._previous[pid] = (ticks, now)
        if previous is None:
            return None
        old_ticks, old_time = previous
        elapsed = now - old_time
        if elapsed <= 0 or ticks < old_ticks:
            return None
        return (ticks - old_ticks) / self._clock_ticks / elapsed * 100.0

    def retain(self, live_pids: Iterable[int]) -> None:
        live = set(live_pids)
        self._previous = {pid: value for pid, value in self._previous.items() if pid in live}


def _match_number(pattern: re.Pattern[str], text: str, group: int = 1):
    match = pattern.search(text)
    return match.group(group) if match else None


def parse_tegrastats(line: str, elapsed_s: float, tag: str) -> BoardSample:
    timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    ram = RAM_RE.search(line)
    swap = SWAP_RE.search(line)
    gpu = GPU_RE.search(line)
    emc = EMC_RE.search(line)
    cpu_match = CPU_RE.search(line)
    power = POWER_RE.search(line)

    cpu_values = (
        [float(value) for value in CPU_ENTRY_RE.findall(cpu_match.group(1))]
        if cpu_match
        else []
    )
    temperatures = {
        name.upper(): float(value)
        for name, value in TEMP_RE.findall(line)
        if float(value) >= 0.0
    }

    gpu_mhz = None
    if gpu and gpu.group(2):
        gpu_mhz = max(int(value) for value in gpu.group(2).split(","))

    return BoardSample(
        timestamp=timestamp,
        elapsed_s=elapsed_s,
        tag=tag,
        gpu_pct=float(gpu.group(1)) if gpu else None,
        gpu_mhz=gpu_mhz,
        emc_pct=float(emc.group(1)) if emc else None,
        emc_mhz=int(emc.group(2)) if emc and emc.group(2) else None,
        cpu_avg_pct=sum(cpu_values) / len(cpu_values) if cpu_values else None,
        cpu_max_pct=max(cpu_values) if cpu_values else None,
        ram_used_mb=int(ram.group(1)) if ram else None,
        ram_total_mb=int(ram.group(2)) if ram else None,
        swap_used_mb=int(swap.group(1)) if swap else None,
        swap_total_mb=int(swap.group(2)) if swap else None,
        gpu_temp_c=temperatures.get("GPU"),
        cpu_temp_c=temperatures.get("CPU"),
        max_temp_c=max(temperatures.values()) if temperatures else None,
        input_power_mw=int(power.group(1)) if power else None,
        input_power_avg_mw=int(power.group(2)) if power else None,
        raw=line.rstrip(),
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except (OSError, PermissionError):
        return ""


def _process_ticks(stat_text: str) -> Optional[int]:
    # /proc/PID/stat field 2 is parenthesized and may contain spaces.
    closing = stat_text.rfind(")")
    if closing < 0:
        return None
    fields = stat_text[closing + 2 :].split()
    try:
        # fields starts at the documented field 3; utime/stime are 14/15.
        return int(fields[11]) + int(fields[12])
    except (IndexError, ValueError):
        return None


def _rss_mb(status_text: str) -> Optional[float]:
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB", status_text, re.MULTILINE)
    return int(match.group(1)) / 1024.0 if match else None


def _gpu_evidence(proc_dir: Path, command: str) -> List[str]:
    evidence = set()
    maps = _read_text(proc_dir / "maps").lower()
    for needle, label in GPU_LIBRARY_HINTS:
        if needle in maps:
            evidence.add(label)

    try:
        for fd in (proc_dir / "fd").iterdir():
            try:
                target = os.readlink(fd).lower()
            except OSError:
                continue
            if "/dev/nvhost-gpu" in target or "/dev/nvidia" in target:
                evidence.add("GPU-device")
                break
    except (OSError, PermissionError):
        pass

    lowered_command = command.lower()
    for needle, label in COMMAND_HINTS:
        if needle in lowered_command:
            evidence.add(label)
    return sorted(evidence)


def scan_gpu_processes(
    tracker: CpuTracker, timestamp: str, elapsed_s: float, tag: str
) -> List[ProcessSample]:
    now = time.monotonic()
    samples: List[ProcessSample] = []
    live_pids: List[int] = []

    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid == os.getpid():
            continue

        raw_cmdline = _read_text(proc_dir / "cmdline").replace("\0", " ").strip()
        command = raw_cmdline or _read_text(proc_dir / "comm").strip()
        if not command:
            continue

        evidence = _gpu_evidence(proc_dir, command)
        if not evidence:
            continue

        ticks = _process_ticks(_read_text(proc_dir / "stat"))
        if ticks is None:
            continue
        live_pids.append(pid)
        samples.append(
            ProcessSample(
                timestamp=timestamp,
                elapsed_s=elapsed_s,
                tag=tag,
                pid=pid,
                cpu_pct=tracker.utilization(pid, ticks, now),
                rss_mb=_rss_mb(_read_text(proc_dir / "status")),
                evidence="+".join(evidence),
                command=command,
            )
        )

    tracker.retain(live_pids)
    return sorted(
        samples,
        key=lambda item: (
            item.cpu_pct is not None,
            item.cpu_pct or 0.0,
            item.rss_mb or 0.0,
        ),
        reverse=True,
    )


def _fmt(value, suffix: str = "", digits: int = 0) -> str:
    if value is None:
        return "?"
    return f"{value:.{digits}f}{suffix}"


def _color_pct(value: Optional[float], text: str, enabled: bool) -> str:
    if not enabled or value is None:
        return text
    code = "31" if value >= 90 else "33" if value >= 70 else "32"
    return f"\033[{code}m{text}\033[0m"


def print_board(sample: BoardSample, color: bool) -> None:
    gpu = _fmt(sample.gpu_pct, "%")
    if sample.gpu_mhz is not None:
        gpu += f"@{sample.gpu_mhz}MHz"
    gpu = _color_pct(sample.gpu_pct, gpu, color)
    emc = _color_pct(sample.emc_pct, _fmt(sample.emc_pct, "%"), color)
    stamp = sample.timestamp[11:19]
    print(
        f"[{stamp}] GPU {gpu:>18} | EMC {emc:>5} | "
        f"CPU avg/max {_fmt(sample.cpu_avg_pct, '%')}/{_fmt(sample.cpu_max_pct, '%')} | "
        f"RAM {_fmt(sample.ram_used_mb, 'MB')}/{_fmt(sample.ram_total_mb, 'MB')} | "
        f"Temp GPU/max {_fmt(sample.gpu_temp_c, 'C', 1)}/{_fmt(sample.max_temp_c, 'C', 1)} | "
        f"Power {_fmt(sample.input_power_mw, 'mW')}"
    )


def print_processes(samples: Sequence[ProcessSample], limit: int) -> None:
    print("  GPU users/candidates (not per-process GPU %):")
    if not samples:
        print("    none visible; run with sudo if other users' /proc entries are hidden")
        return
    print("    PID    CPU%   RSS(MB)  GPU evidence                 command")
    for item in samples[:limit]:
        command = item.command if len(item.command) <= 70 else item.command[:67] + "..."
        cpu = "?" if item.cpu_pct is None else f"{item.cpu_pct:.1f}"
        rss = "?" if item.rss_mb is None else f"{item.rss_mb:.1f}"
        print(f"    {item.pid:<6} {cpu:>6}  {rss:>8}  {item.evidence:<27} {command}")


def percentile(values: Sequence[float], ratio: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(ratio * len(ordered)) - 1))
    return ordered[index]


def print_summary(samples: Sequence[BoardSample]) -> None:
    if not samples:
        return

    def values(name: str) -> List[float]:
        return [float(value) for sample in samples if (value := getattr(sample, name)) is not None]

    print("\nSummary")
    for label, name, suffix in (
        ("GPU", "gpu_pct", "%"),
        ("EMC", "emc_pct", "%"),
        ("CPU average", "cpu_avg_pct", "%"),
        ("Maximum temperature", "max_temp_c", "C"),
        ("Input power", "input_power_mw", "mW"),
    ):
        series = values(name)
        if series:
            print(
                f"  {label:<20} mean={sum(series) / len(series):.1f}{suffix} "
                f"p95={percentile(series, 0.95):.1f}{suffix} max={max(series):.1f}{suffix}"
            )
    gpu = values("gpu_pct")
    if gpu:
        saturated = sum(value >= 90.0 for value in gpu) / len(gpu) * 100.0
        print(f"  GPU >= 90% samples  {saturated:.1f}%")


def csv_writer(path: Optional[Path], fieldnames: Sequence[str]):
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return handle, writer


def dataclass_row(value) -> dict:
    return dict(value.__dict__)


def locate_tegrastats(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    found = shutil.which("tegrastats")
    if found:
        return found
    fallback = Path("/usr/bin/tegrastats")
    return str(fallback) if fallback.exists() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor Jetson Orin GPU/EMC/CPU/temperature/power and GPU-using processes."
    )
    parser.add_argument("--interval", type=int, default=1000, help="sample interval in ms (default: 1000)")
    parser.add_argument("--duration", type=float, default=0.0, help="stop after N seconds; 0 means until Ctrl-C")
    parser.add_argument("--process-every", type=int, default=5, help="show/record GPU processes every N samples")
    parser.add_argument("--process-limit", type=int, default=12, help="maximum processes printed per scan")
    parser.add_argument("--csv", type=Path, help="board CSV path; a sibling *_processes.csv is also written")
    parser.add_argument("--tag", default="", help="experiment label stored in CSV, e.g. all_on or lidar_only")
    parser.add_argument("--raw", action="store_true", help="also print the original tegrastats line")
    parser.add_argument("--no-color", action="store_true", help="disable terminal colors")
    parser.add_argument("--tegrastats", help="explicit tegrastats executable path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval < 100:
        raise SystemExit("--interval must be >= 100 ms")
    if args.process_every < 1:
        raise SystemExit("--process-every must be >= 1")

    executable = locate_tegrastats(args.tegrastats)
    if not executable:
        print("tegrastats was not found. Run this script on the Jetson Orin.", file=sys.stderr)
        return 2

    board_fields = list(BoardSample.__dataclass_fields__)
    process_fields = list(ProcessSample.__dataclass_fields__)
    board_handle, board_csv = csv_writer(args.csv, board_fields)
    process_path = args.csv.with_name(args.csv.stem + "_processes.csv") if args.csv else None
    process_handle, process_csv = csv_writer(process_path, process_fields)

    process = subprocess.Popen(
        [executable, "--interval", str(args.interval)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tracker = CpuTracker()
    samples: List[BoardSample] = []
    start = time.monotonic()
    color = sys.stdout.isatty() and not args.no_color

    def stop_child(_signum=None, _frame=None) -> None:
        if process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGTERM, stop_child)
    signal.signal(signal.SIGINT, stop_child)

    print("Per-process rows identify GPU users, but do not claim per-process GPU percentages.")
    print("Press Ctrl-C to stop and print the summary.\n")
    try:
        assert process.stdout is not None
        for index, line in enumerate(process.stdout, start=1):
            elapsed = time.monotonic() - start
            if "GR3D_FREQ" not in line and "RAM" not in line:
                print(line.rstrip(), file=sys.stderr)
                continue
            sample = parse_tegrastats(line, elapsed, args.tag)
            samples.append(sample)
            print_board(sample, color)
            if args.raw:
                print("  raw:", sample.raw)
            if board_csv:
                board_csv.writerow(dataclass_row(sample))
                board_handle.flush()

            if index == 1 or index % args.process_every == 0:
                processes = scan_gpu_processes(tracker, sample.timestamp, elapsed, args.tag)
                print_processes(processes, args.process_limit)
                if process_csv:
                    process_csv.writerows(dataclass_row(item) for item in processes)
                    process_handle.flush()

            if args.duration > 0 and elapsed >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_child()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if board_handle:
            board_handle.close()
        if process_handle:
            process_handle.close()

    print_summary(samples)
    if args.csv:
        print(f"\nSaved board samples:   {args.csv}")
        print(f"Saved process samples: {process_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
