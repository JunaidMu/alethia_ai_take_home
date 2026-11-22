# Video indexing at speed — YOLO11l over FineVideo

For every second of every video in a folder of FineVideo parquet shards, `run.py` records which of
`person, car, dog, laptop, cell phone` are present and how many people are visible.

## Install (Ubuntu 22.04, Python 3.11, NVIDIA driver with CUDA 12)

```bash
sudo apt-get install -y libgl1 libglib2.0-0     # needed by opencv (imported by ultralytics)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Preparation command (run once, needs internet, not timed)

```bash
python prepare.py        # downloads the official yolo11l.pt next to run.py
```

## Run

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

No internet is needed for the run. Optional flags: `--fps-samples K` (frames run through the model per
second of video, default 3; `0` = every frame), `--workers N` (parallel video decoders), `--batch N`.

## How it works

1. A reader thread opens each parquet file and pulls the `mp4` bytes one row at a time.
2. A pool of decoder threads (PyAV / ffmpeg with frame+slice threading; PyAV releases the GIL) decodes each
   video, groups frames by `floor(pts - first_pts)` seconds and keeps K evenly spaced frames per second.
   The rotation tag is applied, and frames are resized so the long side is 640 (exactly the resize the
   Ultralytics letterbox would do), so far less data reaches the GPU stage.
3. The main thread runs batched FP16 YOLO11l inference (imgsz 640, conf 0.4, default class-aware NMS
   IoU 0.7, BGR input) restricted to the five classes.
4. Per second: an object is present if it is detected in at least half of the sampled frames; people =
   median of the per-frame person counts (rounded down on a half). One JSON line per second is written,
   up to and including the second holding the last frame.
5. Damaged videos/rows produce one `{"video": ..., "error": ...}` line (after any seconds already decoded)
   and processing continues.

`tools/compare.py reference.jsonl candidate.jsonl` scores a result against an every-frame reference with
the metrics from the brief.
