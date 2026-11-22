"""Compare a results.jsonl against a reference (every-frame) results.jsonl
using the brief's metrics: object F1, people exact %, people within-1 %.

    python tools/compare.py reference.jsonl candidate.jsonl
"""
import json
import sys


def load(path):
    d = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "second" in r:
                d[(r["video"], r["second"])] = (set(r["objects"]), r["people"])
    return d


ref, cand = load(sys.argv[1]), load(sys.argv[2])
tp = fp = fn = exact = within = 0
for key, (ro, rp) in ref.items():
    co, cp = cand.get(key, (set(), -99))
    tp += len(ro & co)
    fp += len(co - ro)
    fn += len(ro - co)
    exact += cp == rp
    within += abs(cp - rp) <= 1
n = len(ref)
f1 = 2 * tp / max(1, 2 * tp + fp + fn)
print(f"seconds={n} missing={sum(k not in cand for k in ref)} extra={sum(k not in ref for k in cand)}")
print(f"F1={f1:.4f} (>=0.95)  exact={exact / n:.4f} (>=0.85)  within1={within / n:.4f} (>=0.95)")
