"""Per-second object index for FineVideo parquet shards using YOLO11l.

Usage:
    python run.py --input <folder of .parquet files> --output results.jsonl

Pipeline:
  reader thread  -> reads one parquet row (mp4 bytes) at a time
  decoder pool   -> PyAV decode (ffmpeg frame/slice threads), groups frames by
                    second, keeps K evenly spaced frames per second, applies the
                    rotation tag, resizes so the long side is 640 (what the
                    Ultralytics letterbox would do anyway) and emits BGR chunks
  main thread    -> batched FP16 YOLO11l inference on the GPU, per-second
                    majority vote / median aggregation, JSONL writing
"""

import argparse
import io
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_VERBOSE", "False")

import av
import numpy as np
import pyarrow.parquet as pq

CLASSES = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}
CLASS_IDS = list(CLASSES)
IMGSZ = 640
CONF = 0.4

_END = object()
LETTERBOX = None  # ultralytics LetterBox(640, auto=True, stride=32): identical to the predictor's rect letterbox


def pick_indices(n, k):
    """k evenly spaced indices out of n (all of them if n <= k)."""
    if n <= k:
        return list(range(n))
    return [int((j + 0.5) * n / k) for j in range(k)]


def get_rotation(stream, frame):
    """Clockwise rotation (0/90/180/270) needed to display the frame upright."""
    rot = None
    try:
        r = frame.rotation  # PyAV >= 14: display-matrix angle (counter-clockwise)
        if r:
            rot = (-int(round(r))) % 360
    except AttributeError:
        pass
    if rot is None:
        try:
            tag = stream.metadata.get("rotate")
            if tag is not None:
                rot = int(tag) % 360
        except Exception:
            rot = None
    rot = rot or 0
    return rot - rot % 90


def to_model_input(frame, rot):
    """Rotate + resize so the long side is IMGSZ; returns contiguous BGR uint8."""
    w, h = frame.width, frame.height
    dw, dh = (h, w) if rot in (90, 270) else (w, h)  # displayed size
    r = min(IMGSZ / dw, IMGSZ / dh)
    if r < 1.0:  # shrink here; small frames are left for the letterbox to upscale
        nw, nh = int(round(dw * r)), int(round(dh * r))
    else:
        nw, nh = dw, dh
    # resize in the decoder's native orientation, then rotate the small image
    sw, sh = (nh, nw) if rot in (90, 270) else (nw, nh)
    if (sw, sh) != (w, h):
        img = frame.to_ndarray(width=sw, height=sh, format="bgr24", interpolation="BILINEAR")
    else:
        img = frame.to_ndarray(format="bgr24")
    if rot:
        img = np.rot90(img, k=-(rot // 90))
    return np.ascontiguousarray(img)


def decode_video(key, data, k, out_q, chunk):
    """Decode one video and push sampled frames to out_q.

    Messages: ("frames", key, [second, ...], [img, ...]) and
              ("end", key, n_seconds, error_or_None)
    """
    n_seconds = 0
    error = None
    buf_sec = None
    buf = []
    pend_secs, pend_imgs = [], []
    rot = None

    def flush_second():
        nonlocal pend_secs, pend_imgs
        for i in pick_indices(len(buf), k):
            pend_secs.append(buf_sec)
            pend_imgs.append(LETTERBOX(image=to_model_input(buf[i], rot)))
        if len(pend_imgs) >= chunk:
            out_q.put(("frames", key, pend_secs, pend_imgs))
            pend_secs, pend_imgs = [], []

    try:
        container = av.open(io.BytesIO(data), mode="r")
        try:
            vs = next((s for s in container.streams if s.type == "video"), None)
            if vs is None:
                raise ValueError("no video stream")
            vs.thread_type = "AUTO"
            tb = vs.time_base or Fraction(1, 1000)
            rate = vs.average_rate or vs.guessed_rate or Fraction(30, 1)
            t0 = None
            last_t = None
            for frame in container.decode(vs):
                if frame.pts is not None:
                    t = float(frame.pts * tb)
                elif last_t is not None:
                    t = last_t + float(1 / rate)
                else:
                    t = 0.0
                if t0 is None:
                    t0 = t
                    rot = get_rotation(vs, frame)
                last_t = t
                sec = int((t - t0) // 1) if t >= t0 else 0
                if sec != buf_sec:
                    if buf:
                        flush_second()
                    buf = []
                    buf_sec = sec
                buf.append(frame)
                n_seconds = max(n_seconds, sec + 1)
            if buf:
                flush_second()
                buf = []
        finally:
            container.close()
        if n_seconds == 0:
            error = "no frames decoded"
    except Exception as e:  # damaged / unusual video: keep what we have
        error = f"could not decode video: {type(e).__name__}: {str(e)[:120]}"
        # seconds fully decoded before the failure are kept; the partial one is dropped
        if buf_sec is not None:
            n_seconds = buf_sec
            pend = [(s, im) for s, im in zip(pend_secs, pend_imgs) if s < buf_sec]
            pend_secs = [s for s, _ in pend]
            pend_imgs = [im for _, im in pend]
    if pend_imgs:
        out_q.put(("frames", key, pend_secs, pend_imgs))
    out_q.put(("end", key, n_seconds, error))


def iter_videos(input_dir):
    """Yield (key, mp4_bytes or None, error or None) for every row of every parquet."""
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".parquet"))
    for fname in files:
        path = os.path.join(input_dir, fname)
        try:
            pf = pq.ParquetFile(path)
        except Exception as e:
            yield f"{fname}:0", None, f"could not read parquet: {type(e).__name__}"
            continue
        row = 0
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            try:
                tbl = pf.read_row_group(rg, columns=["mp4"])
                col = tbl.column("mp4")
            except Exception as e:
                for i in range(n):
                    yield f"{fname}:{row + i}", None, f"could not read row: {type(e).__name__}"
                row += n
                continue
            for i in range(n):
                try:
                    v = col[i].as_py()
                except Exception:
                    v = None
                if not v:
                    yield f"{fname}:{row + i}", None, "empty or unreadable mp4"
                else:
                    yield f"{fname}:{row + i}", v, None
            del tbl, col
            row += n


class Aggregator:
    def __init__(self, out_f):
        self.out = out_f
        self.per_video = {}  # key -> {sec: [(people, frozenset(classes)), ...]}
        self.n_videos = 0
        self.n_errors = 0

    def add(self, key, secs, dets):
        d = self.per_video.setdefault(key, {})
        for s, det in zip(secs, dets):
            d.setdefault(s, []).append(det)

    def finish(self, key, n_seconds, error):
        d = self.per_video.pop(key, {})
        lines = []
        for s in range(n_seconds):
            fr = d.get(s, [])
            m = len(fr)
            objs = []
            people = 0
            if m:
                for cid, name in CLASSES.items():
                    hits = sum(1 for _, c in fr if cid in c)
                    if 2 * hits >= m:
                        objs.append(name)
                counts = sorted(p for p, _ in fr)
                if m % 2:
                    people = counts[m // 2]
                else:
                    people = (counts[m // 2 - 1] + counts[m // 2]) // 2
            lines.append(json.dumps({"video": key, "second": s, "objects": objs, "people": people}))
        if error:
            lines.append(json.dumps({"video": key, "error": error}))
            self.n_errors += 1
        if lines:
            self.out.write("\n".join(lines) + "\n")
        self.n_videos += 1



def main():
    global LETTERBOX
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11l.pt"))
    ap.add_argument("--fps-samples", type=int, default=int(os.environ.get("SAMPLES_PER_SEC", 3)),
                    help="frames sampled per second (0 = every frame)")
    ap.add_argument("--workers", type=int, default=max(2, min(8, (os.cpu_count() or 4) // 4)))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--max-videos", type=int, default=0, help="testing: stop after N videos")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import nms

    k = args.fps_samples if args.fps_samples > 0 else 10**9
    device = 0 if torch.cuda.is_available() else "cpu"
    half = bool(args.half) and device == 0
    model = YOLO(args.weights if os.path.exists(args.weights) else "yolo11l.pt")
    # one normal predictor call builds the exact AutoBackend (fused, fp16) that predict() uses
    model.predict([np.zeros((384, 640, 3), np.uint8)] * 2, imgsz=IMGSZ, conf=CONF, iou=0.7, classes=CLASS_IDS,
                  quantize=16 if half else None, device=device, verbose=False, batch=2)
    backend = model.predictor.model
    st = backend.stride
    stride = int(max(st)) if hasattr(st, "__len__") else int(st)
    LETTERBOX = LetterBox((IMGSZ, IMGSZ), auto=True, stride=stride)
    end2end = getattr(backend, "end2end", False)
    dev = model.predictor.device

    @torch.inference_mode()
    def infer(imgs):
        # == BasePredictor.preprocess (images already letterboxed) + inference + DetectionPredictor NMS
        im = torch.from_numpy(np.stack(imgs)).to(dev)
        im = im.permute(0, 3, 1, 2).contiguous().flip(1)
        im = (im.half() if backend.fp16 else im.float()).div_(255)
        preds = backend(im)
        dets = nms.non_max_suppression(preds, CONF, 0.7, CLASS_IDS, False, max_det=300, nc=0, end2end=end2end)
        out = []
        for d in dets:
            cls = d[:, 5].int().tolist()
            out.append((sum(1 for c in cls if c == 0), frozenset(cls)))
        return out

    infer([LETTERBOX(image=np.zeros((360, 640, 3), np.uint8))] * 2)  # warm-up

    out_q = queue.Queue(maxsize=64)
    gpu_q = queue.Queue(maxsize=2)    # batches waiting for the GPU thread
    res_q = queue.Queue()             # results back from the GPU thread
    inflight = threading.Semaphore(args.workers * 2)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    state = {"submitted": 0}

    def task(key, data):
        try:
            decode_video(key, data, k, out_q, args.batch)
        finally:
            inflight.release()

    def reader():
        try:
            for i, (key, data, err) in enumerate(iter_videos(args.input)):
                if args.max_videos and i >= args.max_videos:
                    break
                if err is not None:
                    state["submitted"] += 1
                    out_q.put(("end", key, 0, err))
                    continue
                inflight.acquire()
                state["submitted"] += 1
                pool.submit(task, key, data)
        finally:
            out_q.put(("reader_done",))

    def gpu_worker():
        while True:
            item = gpu_q.get()
            if item is None:
                return
            tags, imgs = item
            try:
                dets = infer(imgs)
            except Exception as e:
                print(f"inference error: {e}", file=sys.stderr)
                dets = [(0, frozenset())] * len(imgs)
            res_q.put((tags, dets))

    t_start = time.time()
    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=gpu_worker, daemon=True).start()

    buckets = {}          # padded shape -> ([(key, sec)], [img])
    pending = {}          # key -> frames sent to the GPU but not back yet
    ended = {}            # key -> (n_seconds, error) once its decoder has finished
    counters = {"finished": 0, "frames": 0}
    reader_done = False

    def dispatch(shape):
        tags, imgs = buckets.pop(shape)
        gpu_q.put((tags, imgs))

    with open(args.output, "w", buffering=1 << 20) as f:
        agg = Aggregator(f)

        def finish(key, n_seconds, error):
            pending.pop(key, None)
            agg.finish(key, n_seconds, error)
            counters["finished"] += 1

        def drain_results(block=False):
            while True:
                try:
                    tags, dets = res_q.get(timeout=0.5) if block else res_q.get_nowait()
                except queue.Empty:
                    return
                block = False
                touched = set()
                for (key, sec), det in zip(tags, dets):
                    agg.add(key, [sec], [det])
                    pending[key] -= 1
                    touched.add(key)
                counters["frames"] += len(tags)
                for key in touched:
                    if pending[key] == 0 and key in ended:
                        finish(key, *ended.pop(key))

        last_log = time.time()
        while True:
            drain_results()
            if reader_done and counters["finished"] >= state["submitted"]:
                break
            try:
                msg = out_q.get(timeout=0.02)
            except queue.Empty:
                msg = None
            if msg is None:
                # decoders are idle: push out partial batches so nothing waits forever
                for shape in list(buckets):
                    dispatch(shape)
                if reader_done and counters["finished"] < state["submitted"]:
                    drain_results(block=True)
                continue
            if msg[0] == "reader_done":
                reader_done = True
            elif msg[0] == "frames":
                _, key, secs, imgs = msg
                pending[key] = pending.get(key, 0) + len(imgs)
                for s, im in zip(secs, imgs):
                    b = buckets.setdefault(im.shape, ([], []))
                    b[0].append((key, s))
                    b[1].append(im)
                    if len(b[1]) >= args.batch:
                        dispatch(im.shape)
            else:
                _, key, n_seconds, error = msg
                if pending.get(key, 0) == 0:
                    finish(key, n_seconds, error)
                else:
                    ended[key] = (n_seconds, error)
            if time.time() - last_log > 30:
                el = time.time() - t_start
                print(f"[{el:7.0f}s] videos {counters['finished']}/{state['submitted']} frames {counters['frames']} "
                      f"({counters['frames'] / el:.0f} fps) errors {agg.n_errors}", file=sys.stderr, flush=True)
                last_log = time.time()
    gpu_q.put(None)
    pool.shutdown(wait=True)
    el = time.time() - t_start
    print(f"done: {counters['finished']} videos, {counters['frames']} frames, {agg.n_errors} errors in {el:.0f}s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
