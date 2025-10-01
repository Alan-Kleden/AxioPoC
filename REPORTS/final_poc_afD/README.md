# Dossier final — PoC AfD

Ce dossier rassemble les artefacts finaux.

## Fichiers clefs
- mini_rapport_final.md — décision et synthèse.
- summary.json, 	op50_messages.csv, bottom50_messages.csv — face-validity message-niveau.
- face_validity_by_thread_* — face-validity par thread (mean/dot significatifs sur extrêmes ; max instable).
- 	emporal_explore_* — exploration des trajectoires (pas de dynamique exploitable).
- axis_stability*.json / *.png — bootstrap (médiane ~0.62).
- holdout_* — hold-out @thread (AUC ≈ 0.50).

## Rejouer minimal
- Face-validity @thread (mean): scripts\face_validity_axis_by_thread.py --agg mean --min_msgs 5
- Hold-out : scripts\eval_thread_holdout.py --agg mean --score cos --min-msgs 5 --test-size 0.2 --seed 0

## Décision
Arrêt du PoC (contrat moral) — voir mini_rapport_final.md.
