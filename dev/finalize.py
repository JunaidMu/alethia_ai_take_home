"""Post-run checks + packaging (dev helper).

1. Verifies results.jsonl.gz: every parquet row has either second lines or an error line,
   and every video's seconds are 0..N-1 with no gaps.
2. Fills the run time into HARDWARE.md from full_run.log.
3. Copies this project's Claude Code transcripts into ai_chats/ with hf_ tokens replaced by [REMOVED].
"""
import glob
import gzip
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# 1. verify results
expected = {}
for p in sorted(glob.glob("finevideo/data/*.parquet")):
    expected[os.path.basename(p)] = pq.ParquetFile(p).metadata.num_rows
secs = defaultdict(set)
errors = {}
n_lines = 0
with gzip.open("results.jsonl.gz", "rt") as f:
    for line in f:
        n_lines += 1
        r = json.loads(line)
        if "error" in r:
            errors[r["video"]] = r["error"]
        else:
            secs[r["video"]].add(r["second"])
want = {f"{fn}:{i}" for fn, n in expected.items() for i in range(n)}
got = set(secs) | set(errors)
missing = want - got
gaps = [v for v, s in secs.items() if s != set(range(max(s) + 1))]
print(f"files={len(expected)} videos expected={len(want)} got={len(got)} missing={len(missing)} "
      f"errors={len(errors)} gaps={len(gaps)} lines={n_lines} seconds={sum(len(s) for s in secs.values())}")
for v, e in list(errors.items())[:20]:
    print("  error:", v, e)
assert not missing, sorted(missing)[:10]
assert not gaps, gaps[:10]

# 2. run time
log = open("full_run.log").read()
start = re.search(r"START (\d+:\d+:\d+)", log).group(1)
end = re.search(r"END (\d+:\d+:\d+)", log).group(1)
dt = datetime.strptime(end, "%H:%M:%S") - datetime.strptime(start, "%H:%M:%S")
mins = dt.seconds // 60
txt = (f"{mins // 60} h {mins % 60:02d} min (started {start}, finished {end} PKT on 22 Nov 2026) for "
       f"{len(want)} videos / {sum(len(s) for s in secs.values())} video-seconds")
hw = open("HARDWARE.md").read().replace("RUN_TIME_PLACEHOLDER", txt)
open("HARDWARE.md", "w").write(hw)
print("HARDWARE.md:", txt)

# 3. AI chats
USER = os.path.basename(os.path.expanduser("~"))  # Windows user-folder name, never hard-coded
src = os.path.expanduser("~/.claude/projects/" + re.sub(r"[^A-Za-z0-9]", "-", ROOT))
os.makedirs("ai_chats/claude_code", exist_ok=True)
tok = re.compile(r"hf_[A-Za-z0-9]{6,}")  # full tokens and partial fragments
# personal identifiers (not code-relevant): email addresses and the Windows user-folder name
personal = [
    (re.compile(r"[A-Za-z0-9._%+-]+@(?!anthropic\.com)[A-Za-z0-9.-]+\.[a-z]{2,}"), "[REMOVED_EMAIL]"),
    (re.compile(re.escape(USER) + r"(?![a-z])"), "[USER]"),
]
try:  # local part of my email (from git config, never hard-coded), e.g. when it appears without the @domain
    import subprocess
    _local = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True).stdout.strip()
    _local = _local.split("@")[0]
    for _part in {_local, re.split(r"[._+-]", _local)[0]}:  # full local part and its name-only prefix
        if len(_part) >= 6:
            personal.append((re.compile(re.escape(_part), re.I), "[REMOVED_EMAIL]"))
except Exception:
    pass
for p in glob.glob(os.path.join(src, "*.jsonl")):
    s = open(p, encoding="utf-8").read()
    s, n = tok.subn("[REMOVED]", s)
    m = 0
    for rx, rep in personal:
        s, k = rx.subn(rep, s)
        m += k
    out = os.path.join("ai_chats/claude_code", os.path.basename(p))
    open(out, "w", encoding="utf-8").write(s)
    print(f"chat {os.path.basename(p)}: {n} token(s) -> [REMOVED], {m} personal identifier(s) redacted")
open("ai_chats/README.md", "w").write(
    "# AI chat history\n\n"
    "`claude_code/` holds the full, unshortened Claude Code session log(s) for this task "
    "(copied from `~/.claude/projects/`).\n\n"
    "Only these edits were made:\n"
    "- Hugging Face access tokens (and partial fragments of one) -> `[REMOVED]`\n"
    "- my email address -> `[REMOVED_EMAIL]`, and my Windows user-folder name in file paths -> `[USER]` "
    "(privacy only; nothing else was changed or removed).\n")
leftover = [p for p in glob.glob("ai_chats/**/*", recursive=True)
            if os.path.isfile(p) and tok.search(open(p, encoding="utf-8", errors="ignore").read())]
assert not leftover, leftover
print("done")
