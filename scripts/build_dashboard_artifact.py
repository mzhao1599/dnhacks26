"""dashboard/index.html + logs/demo.jsonl -> dashboard/artifact.html (the analyst dashboard with the event log
embedded, for publishing). index.html itself keeps fetching ../logs/events.jsonl for local use."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(ROOT, "dashboard", "index.html")).read()
log_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "logs", "demo.jsonl")
SKIP = {"snapshot"}                      # per-episode coach snapshots: 75% of the bytes, unused by the dashboard
lines = []
for l in open(log_path):
    if not l.strip():
        continue
    try:
        if json.loads(l).get("type") in SKIP:
            continue
    except Exception:
        continue                         # torn NFS line
    lines.append(l)
n_ep = 0
for l in lines:
    try:
        n_ep += json.loads(l).get("type") == "episode"
    except Exception:
        pass
label = f"Hopper run 2026-09-05 (all arms, {n_ep:,} episodes)"
head, sep, tail = src.partition('fetch("../logs/events.jsonl")')
assert sep, "index.html fetch line not found"
tail = tail.split("\n", 1)[1]                                  # drop the rest of the fetch line
tail = tail.replace('  .catch(()=>render(sample(),"sample data — open or drop an events.jsonl"));',
                    '  .catch(()=>render(sample(),"sample data — open or drop an events.jsonl"));', 1)
out = head + f'loadText(document.getElementById("fm-data").textContent,{json.dumps(label)})\n' + tail
body = "".join(lines).replace("</script", "<\\/script")
out = out.rstrip() + '\n<script type="application/x-ndjson" id="fm-data">\n' + body + "</script>\n"
open(os.path.join(ROOT, "dashboard", "artifact.html"), "w").write(out)
print("artifact.html", len(out), "bytes,", n_ep, "episodes from", os.path.relpath(log_path, ROOT))
