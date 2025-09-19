# AxioPoC

**Mini-PoC** pour expérimenter les briques de calcul liées à l’axiodynamique :
- `appetition_aversion/` — score net minimal (Fc − Fi)
- `memotion_decay/` — décroissance exponentielle (asymétrique) des mémotions
- `negentropy_telotopic/` — **N_tel** défini par la **mean resultant length** `R ∈ [0,1]`, avec :
  - `ntel_from_radians(...)` (pondérations optionnelles)
  - `ntel_from_degrees(...)` (helper degrés)
  - `ntel_vectorized(...)` (implémentation NumPy vectorisée, support `axis`)
  - `classify_coherence(R, thresholds)` (labels **élevée / moyenne / faible**)

---

## TL;DR

```bash
# (Windows PowerShell) — à la racine du projet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt  # numpy, pytest, etc.

# tests
python -m pytest -q

# CLI (N_tel + label)
python -m negentropy_telotopic --deg 0 10 5 355
python -m negentropy_telotopic --rad 0 3.14159 3.14159 0
python -m negentropy_telotopic --deg 0 0 180 --weights 1 2 3 --high 0.8 --medium 0.6
