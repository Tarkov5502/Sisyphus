"""
tools/stream_bench.py — engine build step 1, measured before a line of C++ exists:
sustained concurrent sequential read across drives, the way a sweep would read them.

Reads large files on each listed drive with several worker processes per drive (each a
plain sequential reader with 8 MB blocks, together giving real queue depth), for a fixed
duration, reporting per-drive and aggregate GB/s every few seconds and at the end. Run it
long enough (5-10 min) to see whether the drives hold their speed under sustained load —
the tape engine reads ~860 GB every sweep for 12 hours.

    python tools/stream_bench.py --seconds 300 --workers 4 ^
        --files "D:\\models\\Kimi-K3-GGUF\\UD-Q2_K_XL\\*.gguf" "C:\\seqread_test.bin"

Each --files argument is a glob; files are grouped by drive letter. Files smaller than the
run will be re-read from the start (wrap-around), which is fine for a bandwidth test but
means C: needs a big enough file set to exceed RAM (32 GB) or the page cache flatters it:
prefer a few of the shards copied to C: over a 12 GB test file.
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

BLOCK = 8 * 1024 * 1024


def worker(paths, seconds, q, wid):
    """Read files round-robin, sequentially, until `seconds` elapse; report bytes per tick."""
    total = 0
    t_end = time.perf_counter() + seconds
    last = time.perf_counter()
    i = wid % max(1, len(paths))
    while time.perf_counter() < t_end:
        p = paths[i % len(paths)]; i += 1
        try:
            with open(p, "rb", buffering=0) as f:
                while time.perf_counter() < t_end:
                    b = f.read(BLOCK)
                    if not b:
                        break
                    total += len(b)
                    now = time.perf_counter()
                    if now - last >= 2.0:
                        q.put((wid, total, now)); last = now
        except OSError as e:
            q.put((wid, -1, str(e))); return
    q.put((wid, total, time.perf_counter()))
    q.put((wid, None, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="globs; grouped by drive letter")
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--workers", type=int, default=4, help="processes per drive")
    a = ap.parse_args()

    by_drive: dict[str, list[str]] = {}
    for g in a.files:
        for p in glob.glob(g):
            if os.path.isfile(p) and os.path.getsize(p) > 64 * 1024 * 1024:
                by_drive.setdefault(os.path.splitdrive(os.path.abspath(p))[0].upper() or "/", []).append(p)
    if not by_drive:
        sys.exit("no files >64 MB matched")
    for d, ps in by_drive.items():
        print(f"{d}: {len(ps)} files, {sum(os.path.getsize(p) for p in ps)/1e9:.1f} GB, {a.workers} workers")

    q = mp.Queue()
    procs = []; owner = {}
    wid = 0
    for d, ps in by_drive.items():
        for k in range(a.workers):
            procs.append(mp.Process(target=worker, args=(ps, a.seconds, q, wid), daemon=True))
            owner[wid] = d; wid += 1
    t0 = time.perf_counter()
    for p in procs:
        p.start()

    latest = {w: 0 for w in owner}           # cumulative bytes per worker
    prev_snapshot = (t0, {d: 0 for d in by_drive})
    done = 0; last_print = t0
    print(f"\n{'t(s)':>6} " + " ".join(f"{d:>10}" for d in by_drive) + f" {'aggregate':>10}   (GB/s over the last interval)")
    while done < len(procs):
        try:
            w, val, ts = q.get(timeout=5.0)
        except Exception:
            continue
        if val is None:
            done += 1; continue
        if val == -1:
            print(f"worker {w} on {owner[w]} failed: {ts}"); done += 1; continue
        latest[w] = val
        now = time.perf_counter()
        if now - last_print >= 5.0:
            cur = {d: 0 for d in by_drive}
            for ww, b in latest.items():
                cur[owner[ww]] += b
            dt = now - prev_snapshot[0]
            rates = {d: (cur[d] - prev_snapshot[1][d]) / dt / 1e9 for d in by_drive}
            print(f"{now - t0:6.0f} " + " ".join(f"{rates[d]:10.2f}" for d in by_drive) + f" {sum(rates.values()):10.2f}")
            prev_snapshot = (now, cur); last_print = now
    for p in procs:
        p.join(timeout=5)
    elapsed = time.perf_counter() - t0
    tot = {d: 0 for d in by_drive}
    for ww, b in latest.items():
        tot[owner[ww]] += b
    print("\nSUSTAINED over %.0f s:" % elapsed)
    for d in by_drive:
        print(f"  {d:6} {tot[d]/1e9:8.1f} GB  {tot[d]/elapsed/1e9:6.2f} GB/s")
    agg = sum(tot.values()) / elapsed / 1e9
    print(f"  aggregate {agg:.2f} GB/s  ->  861 GB sweep = {861/agg:.0f} s at this rate")


if __name__ == "__main__":
    mp.freeze_support()
    main()
