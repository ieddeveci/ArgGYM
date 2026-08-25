import sys, datetime, pathlib

m, lvl, tag = sys.argv[1], sys.argv[2], sys.argv[3]
runs = pathlib.Path("outputs/runs")
suffix = f"__{m}__L{int(lvl):02d}__{tag}"

# Reuse the newest run dir for this cell that has generations but no metrics.json,
# i.e. one whose inference was interrupted. The runner resumes from the sample ids
# already present, so a killed sweep resumes mid-cell instead of discarding it.
# A machine reboot cost 435 of 439 generations this way on 2026-08-24.
candidates = sorted(
    (d for d in runs.glob(f"*{suffix}")
     if d.is_dir() and (d / "generations.jsonl").exists()
     and not (d / "metrics.json").exists()),
    key=lambda d: d.name)
if candidates:
    print(candidates[-1])
else:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(runs / f"{ts}{suffix}")
