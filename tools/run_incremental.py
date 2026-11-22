"""Dev helper used to produce results.jsonl.gz while the dataset was still downloading.

Watches a folder; each time a .parquet file is fully present it runs run.py on it alone
(via a hardlink in a private folder) and writes parts/<name>.jsonl. Resumable.
When `--expect` files are done it concatenates everything into results.jsonl.gz.

    python tools/run_incremental.py finevideo/data out_parts --expect 50 -- --fps-samples 2
"""
import gzip
import os
import shutil
import subprocess
import sys
import time

src, parts = sys.argv[1], sys.argv[2]
expect = int(sys.argv[sys.argv.index("--expect") + 1]) if "--expect" in sys.argv else 50
extra = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(parts, exist_ok=True)
stage = os.path.join(parts, "_stage")

t0 = time.time()
while True:
    names = sorted(f for f in os.listdir(src) if f.endswith(".parquet"))
    todo = [n for n in names if not os.path.exists(os.path.join(parts, n + ".jsonl"))]
    done = len(names) - len(todo)
    if not todo:
        if done >= expect:
            break
        time.sleep(15)
        continue
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    for n in todo:  # every file that is complete right now -> one run.py call (one model load)
        try:
            os.link(os.path.join(src, n), os.path.join(stage, n))
        except OSError:
            shutil.copy(os.path.join(src, n), os.path.join(stage, n))
    tmp = os.path.join(parts, "_batch.tmp")
    t = time.time()
    subprocess.run([sys.executable, os.path.join(here, "run.py"), "--input", stage, "--output", tmp] + extra,
                   check=True)
    outs = {n: [] for n in todo}
    with open(tmp) as f:
        for line in f:
            outs[line.split('"video": "', 1)[1].split(":", 1)[0]].append(line)
    for n, lines in outs.items():
        with open(os.path.join(parts, n + ".part"), "w") as f:
            f.writelines(lines)
        os.replace(os.path.join(parts, n + ".part"), os.path.join(parts, n + ".jsonl"))
    os.remove(tmp)
    print(f"[{time.time() - t0:7.0f}s] {len(todo)} files done in {time.time() - t:.0f}s "
          f"({done + len(todo)}/{expect})", flush=True)

shutil.rmtree(stage, ignore_errors=True)
with gzip.open(os.path.join(here, "results.jsonl.gz"), "wt") as out:
    for n in sorted(os.listdir(parts)):
        if n.endswith(".jsonl"):
            with open(os.path.join(parts, n)) as f:
                shutil.copyfileobj(f, out)
print(f"all done in {time.time() - t0:.0f}s -> results.jsonl.gz")
