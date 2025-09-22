# Benchmark A — Consensus de groupe avec N_tel

## 1) But du benchmark
Montrer que des résumés simples de la cohérence directionnelle d’un fil
— `R_mean`, `R_last`, `R_slope` à partir de la série `R(t)` —
expliquent/prédissent l’issue finale (`outcome` ∈ {0,1}),
et apportent un **complément** par rapport aux baselines « votes bruts ».

---

## 2) Données attendues (CSV)
Colonnes **obligatoires** :
- `thread_id` : identifiant de fil
- `stance` : {−1, 0, +1} (ou libellés équivalents : “support/neutral/oppose”)
- `outcome` : {0,1} (ou “accepted/rejected”)

Colonne **optionnelle** :
- `timestamp` (ISO8601 ou epoch).  
  Si absente → on bascule automatiquement en fenêtrage **par messages**.

Encodage recommandé : UTF-8, séparateur virgule.
- Exemple minimal :
- thread_id,stance,outcome,timestamp
- r1,1,1,2014-01-01T12:34:56Z
- r1,-1,1,2014-01-01T13:00:00Z
- r2,0,0,2014-02-02T08:00:00Z
**Colonne manquante** → le script doit afficher un **message d’erreur clair** et s’arrêter.

---

## 3) Mapping déclaré `stance → θ`
Par défaut (simple et transparent) :
- `s ∈ {−1,0,+1}  →  θ = π·(1−s)/2`  
  `+1→0°`, `0→90°`, `−1→180°`.

**Mapping alternatif (robustesse)** : `θ = arccos(s)`  
→ ne change pas la tendance globale ; sert de contrôle.

---

## 4) Construction de `R(t)` (cohérence directionnelle)
Par fil, on segmente en fenêtres et on calcule le **mean resultant length** :
C = (∑ w_i cos θ_i) / (∑ w_i)
S = (∑ w_i sin θ_i) / (∑ w_i)
R = sqrt(C² + S²) ∈ [0,1]

Fenêtrage au choix :
- **Temporel** : `--time-window 24H` (préférer si `timestamp` fiable)
- **Par messages** : `--msg-window 20` (robuste si dates manquent / sparse)

`R(t)` = suite des `R` par fenêtre.

---

## 5) Features & baselines (par fil)
**Features_R** : `R_mean`, `R_last`, `R_slope` (pente d’une régression linéaire sur `R(t)`).  
**Baselines** :
- `pct_support = # {s=+1} / total`
- `stance_entropy` = entropie de Shannon sur {−1,0,+1}

Jeux testés :
1. `Features_R` = `[R_mean, R_last, R_slope]`
2. `Baselines` = `[pct_support, stance_entropy]`
3. `Combine` = concat des cinq variables

---

## 6) Évaluation propre (anti-circularité)
- Modèle : **LogisticRegression**
- le **fil** (`thread_id`).
- Split : **StratifiedKFold par fil** (évite toute fuite d’info)
- Métriques : **ROC-AUC** (prioritaire), **accuracy** (secondaire)
- Garde-fous :
  - mapping s→θ **déclaré** + **alternatif**
  - deux schémas de fenêtre (24H et 20 messages)
  - **baselines fortes** pour situer la valeur ajoutée de `R`
  - scripts versionnés, paramètres en CLI

---

## 7) Commandes types

### 7.1 Fenêtre 24h (si `timestamp` présent)

python -m benchmarks.rfa_ntel `
  --input data/rfa.csv `
  --time-window 24H `
  --ntel-mode R `
  --plot-mean-rt `
  --save-features out/rfa_features_24h.csv

### 7.2 Par 20 messages (si pas de date / sparse)
  python -m benchmarks.rfa_ntel `
  --input data/rfa.csv `
  --msg-window 20 `
  --ntel-mode cos2 --signed `
  --plot-mean-rt `
  --save-features out/rfa_features_m20.csv

  ### 7.3 Prédiction précoce + « zone grise »
  python -m benchmarks.rfa_ntel `
  --input data/rfa.csv `
  --msg-window 20 `
  --ntel-mode cos2 --signed `
  --early-frac 0.6 `
  --support-band 0.55 0.75 `
  --plot-mean-rt `
  --save-features out/rfa_features_m20_early.csv


