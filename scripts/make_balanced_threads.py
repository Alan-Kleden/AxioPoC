# -*- coding: utf-8 -*-
import csv, os, random, collections
IN = r"data/convokit/cmv/cmv.csv"
OUT = r"data/convokit/cmv/cmv_balanced.csv"
random.seed(42)

by_thread = collections.defaultdict(list)
outcome_by_thread = {}
with open(IN, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        tid = row["thread_id"]
        by_thread[tid].append(row)
        outcome_by_thread[tid] = row["outcome"]

threads0 = [t for t,o in outcome_by_thread.items() if o=="0"]
threads1 = [t for t,o in outcome_by_thread.items() if o=="1"]
n = min(len(threads0), len(threads1))
keep = set(random.sample(threads0, n) + random.sample(threads1, n))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["thread_id","text","timestamp","outcome"])
    w.writeheader()
    for tid in keep:
        for row in by_thread[tid]:
            w.writerow(row)

print(f"OK -> {OUT} (threads 0/1: {n}/{n}, total: {2*n})")
