# AfD — Baseline AxioPoC (VADER)

**Données.** ConvoKit *Wikipedia Articles for Deletion* (383 918 fils, ~3,2 M messages).  
Issue extraite de conversation.meta["outcome_label"]; binarisation **Delete = 1** sinon 0.  
Équilibrage par thread 50/50 → **279 636** fils.  
Fichiers :
- C:\AxioPoC\artifacts_benchAfd_final_vader\features_eval.csv (features alignés)
- C:\AxioPoC\data\wikipedia\afd\afd_eval.csv (labels alignés)

**Features.** R(t) via Benchmarks/text2ntel.py (backend **VADER**), fenêtres **w20** & **w10** → R_mean, R_last, R_slope.  
(Option CMV-like : ajout len_mean, qmark_ratio.)

## Résultats (CV×reps, sets équilibrés 50/50)
| Modèle / Features                         | AUC mean ± sd       | ACC mean ± sd       |
|-------------------------------------------|--------------------:|--------------------:|
| **LogReg** — R_mean_w20                   | **0.601 ± 0.003**   | **0.577 ± 0.003**   |
| **RF** — w20 (R_mean, R_last, R_slope)    | **0.602 ± 0.003**   | **0.573 ± 0.003**   |
| **RF** — w20+w10 (6 colonnes)             | **0.610 ± 0.001**   | **0.580 ± 0.001**   |
| **LogReg** — len_mean, qmark_ratio        | **0.659 ± 0.002**   | **0.615 ± 0.002**   |
| **Sanity (shuffle labels)** — R_mean_w20  | **≈ 0.499 ± 0.002** | **≈ 0.499 ± 0.002** |

**Importances (holdout, RF).**  
AUC holdout = **0.565**.  
Permutation importance (gain moyen) :  
R_mean_w20 (0.0622) > R_last_w20 (0.0557) > R_mean_w10 (0.0443) > R_last_w10 (0.0233) > R_slope_w10 (0.0071) > R_slope_w20 (-0.0011).

## Lecture rapide
- Le signal **R_mean** (valence moyenne) est solide sur AfD ; ajouter w10 améliore légèrement le RF.  
- La baseline **lisible** len_mean + qmark_ratio surpasse les R-features seules (AUC ~**0.659**), cohérent avec CMV.  
- Sanity ≈ **0.50** → pas de fuite.  
- Chaîne stabilisée : IDs harmonisés (int64), intersection features/labels, VADER.

## Prochaines étapes possibles
- Passer les RF en **5×5** pour des écarts-types « définitifs ».  
- Transfert **CMV→AfD** (et inverse) avec --export-model / --import-model.  
- Tester l’apport marginal d’autres features (hedges/politesse, R_iqr, ts_slope_early, etc.).

## Transfert CMV→AfD (généralisation, modèle CMV importé tel quel)

**Setup.** Entraînement sur **CMV** (features communs : `len_mean,qmark_ratio`, LogReg + scaler), import du modèle sur **AfD** sans réentraînement.

**Cmd export (CMV)**
```powershell
python .\scripts\train_export_model.py `
  --feat artifacts_benchB_final\features.csv `
  --raw  data\convokit\cmv\cmv_balanced.csv `
  --cols len_mean,qmark_ratio `
  --model logreg --cv 5 --reps 5 --scale `
  --export-model artifacts_xfer\logreg_lenq_cmv.pkl
```

**Cmd import+test (AfD)**
```powershell
python .\scripts\eval_import_model.py `
  --feat artifacts_benchAfd_final_vader\features_eval_plus.csv `
  --raw  data\wikipedia\afd\afd_eval.csv `
  --cols len_mean,qmark_ratio `
  --import-model artifacts_xfer\logreg_lenq_cmv.pkl
```

**Résultat (réel)** : CMV→AfD (LogReg len+q) = **AUC 0.5003**, **ACC 0.5020** (n = 279 636).
**Conclusion.** Pas de généralisation détectable avec ces 2 features seulement.

## Transfert CMV↔AfD — synthèse rapide

- CMV→AfD (LogReg, 2 cols) : AUC 0.5003, ACC 0.5020 (n=279 636).
- AfD→CMV (RF, 2 cols) : AUC 0.5038, ACC 0.5063 (n=640).
- CMV→AfD (RF, 5 cols proxy) : AUC 0.5011, ACC 0.5019 (n=279 636).
- AfD→CMV (RF, 5 cols proxy) : AUC 0.5057, ACC 0.5172 (n=640).

Conclusion : transfert quasi nul CMV→AfD ; léger signal AfD→CMV (≈0.506).
