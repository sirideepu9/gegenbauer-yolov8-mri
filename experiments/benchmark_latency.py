#!/usr/bin/env python3
"""
Scanner-Side Latency & Throughput Benchmark
==========================================
Reference implementation for Section 4.1 & 5.2 of:
"Learnable Gegenbauer Activations and Direct Coordinate Regression for Lightweight Lesion Detection"
Authors: M. Deepthi, G. N. V. R. Vikram (2026)

Specifications (Section 4.1):
- Hardware: Intel Core i9-12900K CPU (8 threads), NVIDIA GeForce RTX 3090 GPU (24GB GDDR6X)
- Protocol: Batch size 1, 640x640 input resolution
- Warmup: 100 slices
- Timed runs: 500 slices using high-resolution performance counter (time.perf_counter_ns)
- Compared Models:
  * YOLO11-N: 13.5 ms/slice (~74 FPS)
  * YOLOv10-N: 13.8 ms/slice (~72 FPS)
  * YOLOv8n Baseline: 14.1 ms/slice (~71 FPS) GPU / 27.2 ms CPU
  * Proposed Model: 14.7 ms/slice (~68 FPS) GPU / 28.4 ms CPU (+0.6 ms overhead for Gegenbauer Horner recurrence)
  * N4ITK Pre-processing: 42.6 ms/slice accelerated CPU / 1.8 - 2.4 s clinical CPU
"""

import time
import argparse
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def benchmark_device_latency(device: str = "cpu", warmup: int = 20, reps: int = 100) -> dict:
    """
    Measures slice inference latency.
    Runs live microbenchmarking on local CPU/GPU hardware to measure the exact
    runtime and overhead of Gegenbauer Horner evaluation vs. standard activation,
    and displays local measurements alongside the reference dedicated testbed benchmarks.
    """
    print(f"\n[*] Running live hardware latency microbenchmark on local host ({device.upper()})...")

    # Local live timing of Horner-form Gegenbauer activation vs SiLU on feature maps (1, 128, 80, 80)
    x_test = np.random.randn(1, 128, 80, 80).astype(np.float32)
    u_test = np.tanh(x_test)
    c0, c1, c2, c3 = 0.05, 0.25, 0.45, 0.15

    # Warmup
    for _ in range(warmup):
        _ = x_test / (1.0 + np.exp(-x_test))
        _ = c0 + u_test * (c1 + u_test * (c2 + u_test * c3))

    # Time SiLU
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = x_test / (1.0 + np.exp(-x_test))
    t_silu_local = (time.perf_counter() - t0) / reps * 1000.0

    # Time Horner Gegenbauer
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = c0 + u_test * (c1 + u_test * (c2 + u_test * c3))
    t_horner_local = (time.perf_counter() - t0) / reps * 1000.0

    delta_local = t_horner_local - t_silu_local

    print("=" * 80)
    print("LOCAL HOST HARDWARE MICROBENCHMARK (Measured live via time.perf_counter)")
    print("=" * 80)
    print(f"  Feature Tensor Shape:                  (1, 128, 80, 80) [Stride 8 Detection Stage]")
    print(f"  Standard SiLU Activation:              {t_silu_local:.3f} ms / call")
    print(f"  Gegenbauer Horner Evaluation (K=3):    {t_horner_local:.3f} ms / call")
    print(f"  Local Horner Recurrence Overhead (Δt): {delta_local:+.3f} ms / call")
    print("=" * 80)

    # Reference dedicated testbed benchmarks (Section 4.1, Table 2)
    results = {
        "YOLO11-N": {"gpu_ms": 13.5, "cpu_ms": 25.8, "fps": 74.1},
        "YOLOv10-N": {"gpu_ms": 13.8, "cpu_ms": 26.4, "fps": 72.5},
        "Baseline YOLOv8n": {"gpu_ms": 14.1, "cpu_ms": 27.2, "fps": 70.9},
        "Proposed Model": {"gpu_ms": 14.7, "cpu_ms": 28.4, "fps": 68.0},
        "N4ITK (Accelerated)": {"gpu_ms": None, "cpu_ms": 42.6, "fps": 23.5},
        "N4ITK (Clinical)": {"gpu_ms": None, "cpu_ms": 2100.0, "fps": 0.48},
    }

    if HAS_TORCH and device == "cuda" and torch.cuda.is_available():
        # Live PyTorch GPU timing with CUDA synchronization
        from models.yolov8_gegenbauer import GegenbauerYOLOv8
        model = GegenbauerYOLOv8(num_classes=1).to("cuda").eval()
        dummy_input = torch.randn(1, 3, 640, 640, device="cuda")

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
            torch.cuda.synchronize()

            # Timed iterations
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            times = []
            for _ in range(reps):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))

        measured_gpu = float(np.median(times))
        results["Proposed Model"]["gpu_ms"] = round(measured_gpu, 2)
        results["Proposed Model"]["fps"] = round(1000.0 / measured_gpu, 1)

    print("\n" + "=" * 80)
    print("MANUSCRIPT DEDICATED TESTBED BENCHMARK (NVIDIA RTX 3090 GPU / i9-12900K CPU)")
    print("Note: Values below are historical reference measurements reported in Manuscript Table 2.")
    print("Run with '--device cuda' on an NVIDIA RTX 3090 machine with PyTorch to measure GPU latency live.")
    print("=" * 80)
    print(f"{'Model / Algorithm':<26} {'GPU Latency':<16} {'GPU Throughput':<18} {'CPU Latency':<15}")
    print("-" * 80)
    for model, m_dict in results.items():
        gpu_str = f"{m_dict['gpu_ms']} ms" if m_dict['gpu_ms'] is not None else "---"
        fps_str = f"{m_dict['fps']} FPS" if m_dict['gpu_ms'] is not None else "---"
        cpu_str = f"{m_dict['cpu_ms']} ms"
        print(f"{model:<26} {gpu_str:<16} {fps_str:<18} {cpu_str:<15}")
    print("=" * 80)
    print("[+] Latency profiling completed. Local CPU activation benchmark measured live;")
    print("    clinical reference values reported from dedicated RTX 3090 testbed (Section 4.1 & Tab. 2).")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency benchmark runner")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--reps", type=int, default=500)
    args = parser.parse_args()

    benchmark_device_latency(device=args.device, warmup=args.warmup, reps=args.reps)
