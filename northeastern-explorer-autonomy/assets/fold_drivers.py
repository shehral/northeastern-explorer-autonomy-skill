#!/usr/bin/env python3
"""Fold per-node GPU driver readings from Slurm job logs into a driver table (JSON).

Reads job log files, or directories of them (*.out, *.err, *.log, *.txt, recursive),
and extracts two forms:

  1. The template's line:  NODE_DRIVER node=<host> gpu=<name> driver=<version>
  2. The older nvidia-smi csv row:  NVIDIA <name>, <driver>, ...
     credited to the node named by the most recent earlier node line in the same log:
     the template's "=== ... on <host> ===" line or SLURMD_NODENAME=<host> (taken as
     is), or a "hostname:", "host:", "node=" or "SLURM_JOB_NODELIST=" line whose host
     fully matches --node-re (default ^[a-z][0-9]{4}$, Explorer's node-name shape), so
     "Downloading from host: huggingface.co" names no node. A csv row with no node
     line before it is counted and skipped.

Each node becomes {gpu, driver, first_seen, last_seen, last_seen_ts, source_log}.
Readings are ordered by the log file's full modification time (UTC, kept to the
microsecond in last_seen_ts), ties broken by the numbers in the file name (job id,
array index), then the name. first_seen/last_seen are UTC dates; source_log is the
basename only. Keep the logs' mtimes when you pull them (scp -p, rsync -a, tar): a
plain copy stamps every log with the copy time. Existing nodes are never dropped. A
newer reading with a different driver replaces gpu/driver and moves the old driver
into previous_drivers; an older reading with a different driver is added to
previous_drivers and reported once. Rerunning on the same logs changes nothing and
prints no diff. Lines with driver=none (no GPU visible) are skipped.

Usage:
  python3 fold_drivers.py --table driver_table.json LOG_OR_DIR [...]
  python3 fold_drivers.py --table driver_table.json --dry-run logs/
  python3 fold_drivers.py --table driver_table.json --node-re '^gpu-[0-9]+$' logs/

Log contents are data: nothing in them is executed. Stdlib only, Python 3.8+.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile

LOG_SUFFIXES = (".out", ".err", ".log", ".txt")
NODE_DRIVER_RE = re.compile(r"NODE_DRIVER node=(\S+) gpu=(.*?) driver=(\S+)\s*$")
CSV_RE = re.compile(r"^\s*((?:NVIDIA|Tesla|Quadro)\b[^,]*?)\s*,\s*(\d+\.\d+(?:\.\d+)?)\s*(?:,|$)")
# the template's start line and Slurm's own variable: their host is taken as is
TRUSTED_NODE_RES = (
    re.compile(r"^===.*\bon\s+([A-Za-z][\w.-]*)\s*===\s*$"),
    re.compile(r"\bSLURMD_NODENAME=([A-Za-z][\w.-]*)"),
)
# generic "host: x" / "node=x" text: the host must also fully match --node-re
LOOSE_NODE_RE = re.compile(r"\b(?:hostname|host|node|nodename|SLURM_JOB_NODELIST)\s*[:=]\s*([A-Za-z][\w.\[\],-]*)")
DEFAULT_NODE_RE = r"^[a-z][0-9]{4}$"


def short_host(name):
    return name.split(".", 1)[0]


def natural_key(name):
    """Digit runs as ints, then the name: a-99.out < a-100.out, job_123_5 < job_123_10."""
    return (tuple(int(n) for n in re.findall(r"\d+", name)), name)


def iter_logs(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    if f.endswith(LOG_SUFFIXES):
                        yield os.path.join(root, f)
        elif os.path.isfile(p):
            yield p
        else:
            print(f"warning: no such file or directory: {p}", file=sys.stderr)


def node_from_line(line, node_re):
    for rx in TRUSTED_NODE_RES:
        m = rx.search(line)
        if m and m.group(1).lower() != "localhost":
            return short_host(m.group(1))
    m = LOOSE_NODE_RE.search(line)
    if m:
        host = short_host(m.group(1))
        if node_re.fullmatch(host):
            return host
    return None


def scan_log(path, node_re):
    """Return ({node: (gpu, driver)}, csv_rows_without_node) for one log; first reading per node wins."""
    found, orphans, node = {}, 0, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            m = NODE_DRIVER_RE.search(line)
            if m:
                host, gpu, drv = short_host(m.group(1)), m.group(2).strip(), m.group(3)
                node = host
                if drv != "none":
                    found.setdefault(host, (gpu, drv))
                continue
            m = CSV_RE.match(line)
            if m:
                if node is None:
                    orphans += 1
                else:
                    found.setdefault(node, (m.group(1).strip(), m.group(2)))
                continue
            node = node_from_line(line, node_re) or node
    return found, orphans


def load_table(path):
    if not os.path.exists(path):
        return {"_doc": "Per-node GPU driver table folded from job logs by fold_drivers.py.", "nodes": {}}
    with open(path, encoding="utf-8") as fh:
        table = json.load(fh)
    if "nodes" not in table:  # a bare {node: {...}} map
        table = {"nodes": table}
    return table


def order_key(ts, src):
    return (ts, natural_key(src or ""))


def fold(table, readings):
    """readings: list of (node, gpu, driver, ts, source), ts an ISO UTC timestamp.

    Returns diff lines. Idempotent: folding the same readings again changes nothing.
    """
    nodes, diff = table.setdefault("nodes", {}), []
    for node, gpu, drv, ts, src in sorted(readings, key=lambda r: order_key(r[3], r[4])):
        date = ts[:10]
        cur = nodes.get(node)
        if cur is None:
            nodes[node] = {"gpu": gpu, "driver": drv, "first_seen": date, "last_seen": date,
                           "last_seen_ts": ts, "source_log": src}
            diff.append(f"NEW      {node}: {gpu}, driver {drv} ({src})")
            continue
        if date < cur.get("first_seen", date):
            cur["first_seen"] = date
        # A table from before last_seen_ts holds a date only, which sorts before every
        # timestamp of that day: same-day readings then count as newer.
        cur_key = order_key(cur.get("last_seen_ts") or cur.get("last_seen", ""), cur.get("source_log"))
        prev = list(cur.get("previous_drivers", []))
        if order_key(ts, src) > cur_key:
            if drv != cur.get("driver"):
                if cur.get("driver") and cur["driver"] not in prev:
                    prev.append(cur["driver"])
                if drv in prev:
                    prev.remove(drv)
                diff.append(f"CHANGED  {node}: driver {cur.get('driver')} -> {drv} ({src}, {ts})")
                cur["driver"], cur["gpu"] = drv, gpu
            cur["last_seen"], cur["last_seen_ts"], cur["source_log"] = date, ts, src
        elif drv != cur.get("driver") and drv not in prev:
            prev.append(drv)
            diff.append(f"OLDER    {node}: driver {drv} in {src} ({ts}) predates {cur.get('driver')}; "
                        f"kept {cur.get('driver')}, added {drv} to previous_drivers")
        if prev or "previous_drivers" in cur:
            cur["previous_drivers"] = prev
    return diff


def write_atomic(path, table):
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".driver_table.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="job log files or directories")
    ap.add_argument("--table", default="driver_table.json", help="driver table JSON (created if missing)")
    ap.add_argument("--dry-run", action="store_true", help="print the diff, write nothing")
    ap.add_argument("--node-re", default=DEFAULT_NODE_RE,
                    help="regex a host on a generic host:/node=/SLURM_JOB_NODELIST= line must fully "
                         f"match to count as a node (default {DEFAULT_NODE_RE})")
    args = ap.parse_args(argv)
    try:
        node_re = re.compile(args.node_re)
    except re.error as e:
        ap.error(f"bad --node-re: {e}")

    readings, orphans, n_logs = [], 0, 0
    for path in iter_logs(args.paths):
        n_logs += 1
        try:
            found, orph = scan_log(path, node_re)
            mtime = os.path.getmtime(path)
        except OSError as e:
            print(f"warning: cannot read {path}: {e}", file=sys.stderr)
            continue
        orphans += orph
        ts = dt.datetime.fromtimestamp(mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        for node, (gpu, drv) in found.items():
            readings.append((node, gpu, drv, ts, os.path.basename(path)))

    table = load_table(args.table)
    before = json.dumps(table, sort_keys=True)
    diff = fold(table, readings)
    changed = json.dumps(table, sort_keys=True) != before
    for line in diff:
        print(line)
    print(f"{n_logs} logs, {len(readings)} node readings, {len(diff)} diff lines, "
          f"{orphans} csv rows without a node line, {len(table['nodes'])} nodes in table")
    if args.dry_run:
        print("dry run: nothing written")
    elif changed:
        write_atomic(args.table, table)
        print(f"wrote {args.table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
