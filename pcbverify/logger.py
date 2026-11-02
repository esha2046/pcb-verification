"""Inspection logging for analytics (JSON lines + CSV) - last box of the workflow slide."""
import csv, json, os, time

def log_result(summary, components, source, path="out/inspection_log.csv", jsonl="out/inspection_log.jsonl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source, **summary}
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); 
        if new: w.writeheader()
        w.writerow(row)
    with open(jsonl, "a") as f:
        f.write(json.dumps({**row, "components": [{k: c[k] for k in ("name", "cls", "status", "conf")} for c in components]}) + "\n")
