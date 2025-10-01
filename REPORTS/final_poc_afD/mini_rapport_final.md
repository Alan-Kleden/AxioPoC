# PoC AfD/CMV — Rapport final (résumé)

**Décision : arrêt du PoC (contrat moral)**
- AUC hold-out @thread < 0.55 (≈ 0.50 observé)
- Early→Late non probant (Δpp et p-value insuffisants)
- Gain vs baselines < 0.02

## 1) Question
Valider un axe télotopique exploitable (pro-Delete ↔ pro-Keep) et, si possible, une dynamique temporelle prédictive. Tester transfert vers CMV.

## 2) Données
- AfD : 1 167 050 msgs ; threads éligibles (≥5 msgs, labellés) = 66 175.
- Textes filtrés (dépouillés HTML/wiki/jargon) : ~1.21 M.
- CMV : structure non séquentielle (≈1 msg/fil), inadaptée au temporel.

## 3) Méthodes
- Axe μ₊−μ₋ (Delete−Keep) sur embeddings message (MiniLM 384d).
- Scores msg (dot/cos) → agrégation thread (mean/max).
- Face-validity (tri extrêmes + χ²), bootstrap stabilité, early-only (non-circularité),
  hold-out @thread (80/20), SAFE perms.

## 4) Résultats (synthèse)
- Face-validity @thread (mean) : signal **faible mais significatif** sur extrêmes (p~2.6e-4, OR~1.6, Δpp~+12).
- Agrégat **max** : instable (inversion) → à proscrire.
- Stabilité axe (B=1000) : médiane cos ~0.62 (**insuffisant** vs seuil 0.90).
- Non-circularité early-only (30–50%) : **non robuste**.
- Hold-out @thread (80/20, mean/cos) : **AUC ≈ 0.50** (échec de généralisation).

## 5) Conclusion
Signal **statique** faible au niveau “orientation moyenne de thread”, mais **aucune généralisation**
et **pas de structure temporelle** exploitable. Arrêt du PoC conforme au contrat.

## 6) Valeur méthodo
Pipelines reproductibles (face-validity, bootstrap, hold-out, ablations confondeurs).
Résultats négatifs propres et documentés.

## 7) Reprise éventuelle (hors présent PoC)
Corpus fortement séquentiel, axe exogène (seedé), validation qualitative stricte,
modèles contextuels conversationnels, benchmarks vs baselines simples.
