# tests/test_benchmark_a_smoke.py
import csv
import os
import sys
import subprocess
import pathlib

def _write_toy_csv(path: pathlib.Path):
    rows = [
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
    # Répertoire racine du repo (tests/ -> ..)
    repo_root = pathlib.Path(__file__).resolve().parents[1]

    # Données & artefacts CSV dans un dossier temporaire isolé
    workdir = tmp_path
    data_csv = workdir / "toy.csv"
    out_tmp = workdir / "out"
    out_tmp.mkdir(parents=True, exist_ok=True)
    _write_toy_csv(data_csv)

    # Le script enregistre la figure dans "out/mean_rt.png" RELATIF AU CWD
    # On lance donc le sous-processus depuis la racine du repo (import OK, chemins relatifs connus)
    expected_plot = repo_root / "out" / "mean_rt.png"
    if expected_plot.exists():
        expected_plot.unlink()  # nettoyage si un ancien run a laissé un fichier

    cmd = [
        sys.executable, "-m", "benchmarks.rfa_ntel",
        "--input", str(data_csv),
        "--msg-window", "2",
        "--ntel-mode", "R",
        "--plot-mean-rt",
        "--save-features", str(out_tmp / "features.csv"),
    ]
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),           # <<< racine du repo (module importable)
        timeout=120,
    )
    assert result.returncode == 0, f"Process failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Artefacts attendus
    assert (out_tmp / "features.csv").exists(), "features.csv is missing"
    assert expected_plot.exists(), "mean_rt.png is missing"

    # Nettoyage léger du plot pour ne pas salir le repo local/CI
    try:
        expected_plot.unlink()
    except Exception:
        pass
