import sys, subprocess, re

def run_cli(args):
    cmd = [sys.executable, "-m", "negentropy_telotopic"] + args
    out = subprocess.check_output(cmd, text=True).strip()
    return out

def extract_R(output: str) -> float:
    m = re.search(r"R\s*=\s*([0-9.]+)", output)
    assert m, f"Impossible d'extraire R depuis: {output}"
    return float(m.group(1))

def extract_label(output: str) -> str:
    m = re.search(r"coherence\s*=\s*(\w+)", output)
    assert m, f"Impossible d'extraire le label depuis: {output}"
    return m.group(1)

def test_cli_degrees_high():
    out = run_cli(["--deg", "0", "10", "5", "355"])
    R = extract_R(out); label = extract_label(out)
    assert R > 0.9 and label == "élevée"

def test_cli_radians_low():
    out = run_cli(["--rad", "0", "3.14159", "3.14159", "0"])
    R = extract_R(out); label = extract_label(out)
    assert R < 1e-3 and label == "faible"

def test_cli_custom_thresholds_medium():
    # seuils personnalisés: high=0.95, medium=0.5
    # Choix d'angles symétriques ±30° autour de 0° :
    # R = (1 + 2 cos 30°)/3 ≈ 0.91068 -> "moyenne"
    out = run_cli(["--deg", "0", "30", "330", "--high", "0.95", "--medium", "0.5"])
    R = extract_R(out); label = extract_label(out)
    assert 0.5 <= R < 0.95 and label == "moyenne"

