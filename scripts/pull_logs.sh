#!/bin/bash
# Pull every Hopper event log back and merge the demo-relevant ones into logs/demo.jsonl (dashboard input).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/hopper
# Project dir on the cluster; defaults to /scratch/<your cluster user>/fleet-memory.
REMOTE_ROOT=${FM_REMOTE_ROOT:-$(ssh hopper 'echo /scratch/$USER/fleet-memory')}
rsync -aq --include='*/' --include='*.jsonl' --include='*.png' --exclude='*' "hopper:${REMOTE_ROOT}/logs/" logs/hopper/
python3 - <<'PY'
import glob, json, os
parts = ["logs/hopper/phase0/events.jsonl", "logs/hopper/probe_v31/*.jsonl", "logs/hopper/v31/events.jsonl",
         "logs/hopper/benchmark/events.jsonl", "logs/hopper/bench_camera/*.jsonl", "logs/hopper/protocol/events.jsonl", "logs/hopper/armC/events.jsonl"]
n = 0
with open("logs/demo.jsonl", "w") as out:
    for pat in parts:
        for p in sorted(glob.glob(pat)):
            for line in open(p):
                if line.strip():
                    out.write(line if line.endswith("\n") else line + "\n"); n += 1
print(f"logs/demo.jsonl: {n} events from {sum(len(glob.glob(p)) for p in parts)} files")
PY
