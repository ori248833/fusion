# Orin GPU monitoring

Run on the Jetson Orin from the repository root:

```bash
python3 tools/orin_gpu_watch.py --duration 60 --tag all_on \
  --csv logs/orin_all_on.csv
```

The terminal shows board-wide GPU (`GR3D_FREQ`), memory-controller load
(`EMC_FREQ`), CPU, RAM, temperature and total input power. Every five samples it
also lists processes that opened NVIDIA GPU devices or loaded CUDA, TensorRT,
cuDNN, ZED, or graphics libraries.

The process list deliberately does **not** show a per-process GPU percentage.
`tegrastats` does not expose a reliable percentage for each process. CPU and RSS
are displayed only as supporting evidence. Run the script with `sudo` if Linux
permissions hide `/proc` data belonging to other users.

## Recommended comparison

Collect the same 60-second steady-state window for each condition:

```bash
python3 tools/orin_gpu_watch.py --duration 60 --tag idle \
  --csv logs/orin_idle.csv
python3 tools/orin_gpu_watch.py --duration 60 --tag sensors_only \
  --csv logs/orin_sensors_only.csv
python3 tools/orin_gpu_watch.py --duration 60 --tag lidar_only \
  --csv logs/orin_lidar_only.csv
python3 tools/orin_gpu_watch.py --duration 60 --tag camera_only \
  --csv logs/orin_camera_only.csv
python3 tools/orin_gpu_watch.py --duration 60 --tag all_on \
  --csv logs/orin_all_on.csv
```

Use the difference between conditions to estimate each pipeline's marginal GPU
and EMC cost. Keep sensor rates, scene, power mode and warm-up time consistent.

## Interactive process view

If `jetson-stats` is already installed, `jtop` is a convenient interactive
companion. Its GPU page lists GPU processes and their GPU memory use:

```bash
jtop
```

GPU memory is not GPU execution time. A process can reserve a large TensorRT
engine but be mostly idle, or use little memory while frequently launching GPU
kernels.

## Exact process attribution

For a short diagnostic run, profile the complete ROS launch with Nsight Systems:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=process-tree \
  --duration=20 --output=fusion_gpu ros2 launch my_launch system_run.launch.py
```

Open `fusion_gpu.nsys-rep` in Nsight Systems and inspect the CUDA contexts and
streams below each process. Check `nsys profile --help` on the Orin first,
because the Nsight version bundled with different JetPack releases can expose
slightly different options. Profiling adds overhead, so use it for diagnosis,
not for final frame-rate measurements.
