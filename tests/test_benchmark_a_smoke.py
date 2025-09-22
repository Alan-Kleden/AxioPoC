# tests/test_benchmark_a_smoke.py
import csv
import os
import sys
import subprocess
import pathlib

def _write_toy_csv(path: pathlib.Path):
    rows = [
        # thread_id, stance, outcome, timestamp
        ("t1",  1, 1, "2024-01-01T12:00:00Z"),
        ("t1",  1, 1, "2024-01-01T13:00:00Z"),
        ("t1", -1, 1, "2024-01-01T14:00:00Z"),
        ("t2", -1, 0, "2024-01-02T09:00:00Z"),
        ("t2",  0, 0, "2024-01-02T10:00:00Z"),
        ("t2",  1, 0, "2024-01-02T11:00:00Z"),
        ("t3",  1, 1, "2024-01-03T10:00:00Z"),
        ("t3",  1, 1, "2024-01-03T11:00:00Z"),
        ("t3",  1, 1, "2024-01-03T12:00:00Z"),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["thread_id", "stance", "outcome", "timestamp"])
        w.writerows(rows)

def test_benchmark_a_end_to_end(tmp_path: pathlib.Path):
    # On travaille dans un dossier isolé pour ne rien salir
    workdir = tmp_path
    data_csv = workdir / "toy.csv"
    out_dir = workdir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_toy_csv(data_csv)

    cmd = [
        sys.executable, "-m", "benchmarks.rfa_ntel",
        "--input", str(data_csv),
        "--msg-window", "2",           # petites fenêtres pour générer plusieurs points
        "--ntel-mode", "R",
        "--plot-mean-rt",
        "--save-features", str(out_dir / "features.csv"),
    ]
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    # IMPORTANT : on exécute dans workdir pour que "out/mean_rt.png" sorte là-bas
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=str(workdir), timeout=90
    )
    assert result.returncode == 0, f"Process failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Artefacts attendus
    assert (out_dir / "features.csv").exists(), "features.csv is missing"
    assert (out_dir / "mean_rt.png").exists(), "mean_rt.png is missing"
