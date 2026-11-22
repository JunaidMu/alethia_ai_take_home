# Hardware used for the full 50-file run

**Where:** my own laptop — Lenovo Legion Pro 5 16IAX10 (Legion Space "Performance" power mode, on AC power).

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB GDDR7 (115 W TGP), driver 617.42, CUDA 12.8 PyTorch build |
| CPU | Intel Core Ultra 9 275HX — 24 cores (8P + 16E) / 24 threads, up to 5.4 GHz |
| RAM | 32 GB DDR5-5600 (2 × 16 GB) |
| Disk | 1 TB M.2 PCIe 4.0 x4 NVMe SSD |
| OS | Windows 11 Home (Python 3.12, torch 2.11.0+cu128, ultralytics 8.4.216, PyAV 19.0.2) |

## Full run time

- Settings: `--fps-samples 3` (the default), FP16, batch 64.
- Start to finish: **RUN_TIME_PLACEHOLDER**
- The run was overlapped with the dataset download (files were processed as soon as each
  parquet finished downloading, via `tools/run_incremental.py`, which calls `run.py`), so the wall time
  above is GPU-bound inference time, not download time. Sustained throughput was ~143 frames/s through
  YOLO11l on this laptop GPU.
