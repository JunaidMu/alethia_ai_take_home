# Hardware used for the full 50-file run

**Where:** my own laptop — Lenovo Legion Pro 5 16IAX10 (Legion Space "Performance" power mode, on AC power).

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB GDDR7 (115 W TGP), driver 617.42, CUDA 12.8 PyTorch build |
| CPU | Intel Core Ultra 9 275HX — 24 cores (8P + 16E) / 24 threads, up to 5.4 GHz |
| RAM | 32 GB DDR5-5600 (2 × 16 GB) |
| Disk | 1 TB M.2 PCIe 4.0 x4 NVMe SSD |
| OS | Windows 11 Home (Python 3.12, torch 2.11.0+cu128, ultralytics 8.4.216, PyAV 19.0.2, numpy 2.5.4; `requirements.txt` pins av 18.1.1 / numpy 2.4.6 because the newer ones have no Python 3.11 wheels — numpy 2.4.6 re-verified identical on the validation videos) |

## Full run time

- Settings: `--fps-samples 3` (the default), FP16, batch 64.
- Start to finish: **2 h 11 min (started 21:00:45, finished 23:12:12 PKT on 22 Nov 2026) for 1650 videos / 461757 video-seconds**
- 1,385,035 sampled frames went through YOLO11l, at a sustained ~176 frames/s (GPU 91–100 % busy,
  ~6.5 GB of 8 GB VRAM in use). 0 videos failed.
- The run started while the dataset was still downloading (the download finished at ~21:25), using
  `tools/run_incremental.py`, which calls `run.py` on whichever parquet files are complete. It made two
  `run.py` calls (27 files in 4,295 s, then 23 files in 3,580 s), so the time above includes two model
  start-ups (~20 s each) and building the gzip at the end.
