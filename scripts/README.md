# Scripts — AxioPoC

Ce dossier regroupe les utilitaires pour (1) reconstruire les données AfD depuis ConvoKit,
(2) fabriquer les features “axio” (pragmatiques + R* proxies), et (3) évaluer la Distance
axiologique (Dₐ) en intra-corpus et en transfert CMV↔AfD.

> Prérequis : venv actif, se placer à la racine du repo (`C:\AxioPoC`), Python 3.10+,
> `pandas`, `scikit-learn`, `joblib`.

---

## 1) Reconstruction AfD (ConvoKit → CSV)

- `afd_rebuild_from_convokit.py`
  - Entrée : ConvoKit `wiki-articles-for-deletion-corpus` (dans `~/.convokit/saved-corpora/...`)
  - Sorties :
    - `artifacts_xfer/afd_messages.csv` (thread_id, text)
    - `artifacts_xfer/afd_idmap.csv` (conversation_id → thread_id)
  - Usage :
    ```powershell
    python scripts/afd_rebuild_from_convokit.py
    ```

- `afd_select_labeled_messages.py`
  - Ajoute la colonne `label` (0/1) aux messages AfD selon le mapping existant.
  - Sortie : `artifacts_xfer/afd_messages_labeled.csv`
  - Usage :
    ```powershell
    python scripts/afd_select_labeled_messages.py
    ```

---

## 2) Features “axio” (CMV / AfD)

- `make_features_axio.py`
  - Fusionne les colonnes nécessaires et crée les **proxies R*** :
    - CMV : `R_proxy=R_mean`, `R_last_proxy=R_last`, `R_slope_proxy=R_slope`
    - AfD : `R_proxy=R_mean_w20`, `R_last_proxy=R_last_w20`, `R_slope_proxy=R_slope_w20`
  - Usage :
    ```powershell
    # CMV
    python scripts/make_features_axio.py --in artifacts_xfer/features_cmv_lenqR.csv --out artifacts_xfer/features_axio_cmv.csv
    # AfD
    python scripts/make_features_axio.py --in artifacts_xfer/features_afd_lenqR.csv --out artifacts_xfer/features_axio_afd.csv
    ```

- `afd_build_r_slope.py` / `afd_fill_r_slope_from_windows.py`
  - Complètent/recoupent `R_slope_proxy` depuis `rt_series.csv` (ou `features_eval_plus.csv`).
  - Usage :
    ```powershell
    python scripts/afd_build_r_slope.py --series artifacts_benchAfd_vader_w20/rt_series.csv `
      --axio-in artifacts_xfer/features_axio_afd.csv --axio-out artifacts_xfer/features_axio_afd.csv
    ```

- `make_pragmatics_afd.py`
  - Extrait marqueurs pragmatiques (ratios) depuis `afd_messages_labeled.csv`
    et les merge dans `features_axio_afd.csv`.
  - Usage :
    ```powershell
    python scripts/make_pragmatics_afd.py --in artifacts_xfer/afd_messages_labeled.csv `
      --text-col text --out-feat artifacts_xfer/features_axio_afd.csv
    ```

---

## 3) Signatures Dₐ et évaluations

- `make_signature.py`
  - Sauvegarde la **signature** (liste ordonnée des colonnes) dans un `.pkl`.
  - Exemple (10 colonnes) :
    ```powershell
    $COL10 = "len_mean,qmark_ratio,R_proxy,R_last_proxy,R_slope_proxy,polite_ratio,hedge_ratio,you_i_ratio,agree_markers,neg_markers"
    python scripts/make_signature.py --feat artifacts_xfer/features_axio_cmv.csv --out artifacts_xfer/signature_cmv_10.pkl --cols $COL10
    python scripts/make_signature.py --feat artifacts_xfer/features_axio_afd.csv --out artifacts_xfer/signature_afd_10.pkl --cols $COL10
    ```

- `eval_da.py` (intra-corpus)
  - Calcule Dₐ (métrique diag ou Σ⁻¹), CV LogReg, AUC/ACC.
  - Exemple :
    ```powershell
    python scripts/eval_da.py --feat artifacts_xfer/features_axio_cmv.csv --raw data/convokit/cmv/cmv_balanced.csv `
      --signature artifacts_xfer/signature_cmv_10.pkl --metric diag --cv 5 --reps 3
    ```

- `eval_da_transfer.py` (transfert source→cible)
  - Apprend sur la **source**, évalue sur la **cible** (même signature).
  - Exemple :
    ```powershell
    python scripts/eval_da_transfer.py --feat-src artifacts_xfer/features_axio_cmv.csv --raw-src data/convokit/cmv/cmv_balanced.csv `
      --signature-src artifacts_xfer/signature_cmv_10.pkl --feat-tgt artifacts_xfer/features_axio_afd.csv `
      --raw-tgt data/wikipedia/afd/afd_eval.csv --signature-tgt artifacts_xfer/signature_afd_10.pkl --metric diag
    ```

- `learn_da_weights.py` (optionnel)
  - Apprend des **poids diag(w)** par dimension (pondération de Dₐ).
  - Exemple :
    ```powershell
    python scripts/learn_da_weights.py --feat artifacts_xfer/features_axio_cmv.csv --raw data/convokit/cmv/cmv_balanced.csv `
      --cols $COL10 --out artifacts_xfer/weights_cmv_10.pkl
    ```

---

## Bonnes pratiques

- **Imputation** : si des NaN apparaissent dans les colonnes pragmatiques, imputer à **0.0**.
- **Robustesse** (option code) : dans `eval_da*.py`, avant l’entraînement,
  ```python
  import numpy as np
  Da = np.nan_to_num(Da, nan=0.0, posinf=0.0, neginf=0.0)


---

## À quoi sert la ligne `.gitignore` “consolidée” ?

Elle **dit à Git de ne pas suivre** certains fichiers/dossiers générés localement
(artefacts, données, backups). Résultat :
- le repo reste **léger** et **propre** ;
- pas de **binaires lourds** ou **fichiers temporaires** dans l’historique ;
- `git status` n’est pas “pollué”.

### Bloc `.gitignore` conseillé (prêt-à-coller à la racine)

Données & artefacts lourds

data/
artifacts_*/
artifacts_xfer/
*.pkl
*.npz
*.npy
*.log

Backups locaux & caches

data_bak_*/
*.tmp
*.bak
.DS_Store
Thumbs.db

Environnements / IDE

.venv/
venv/
.env
.ipynb_checkpoints/
.vscode/