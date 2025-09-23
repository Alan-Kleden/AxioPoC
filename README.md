![ci](https://github.com/Alan-Kleden/AxioPoC/actions/workflows/ci.yml/badge.svg)

# AxioPoC

**Mini-PoC** pour expérimenter les briques de calcul liées à l’axiodynamique :
- `appetition_aversion/` — score net minimal (Fc − Fi)
- `memotion_decay/` — décroissance exponentielle (asymétrique) des mémotions
- `negentropy_telotopic/` — cohérence directionnelle **N_tel**
  - `ntel_from_radians(...)` (mode **R**, "mean resultant length", robuste, [0..1])
  - `ntel_cos2_from_radians(...)` (mode **cos2**, autour d’un télos explicite ou endogène)
  - `ntel_entropy_from_radians(...)` (mode **entropy**, \(N = 1 - H\) normalisé)
  - `ntel_from_degrees(...)`, `ntel_vectorized(...)`, `classify_coherence(...)`
- `benchmarks/` — **Benchmark A (RfA)** : consensus de groupe avec N_tel

## Démarrage rapide (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .[test]
python -m pytest -q