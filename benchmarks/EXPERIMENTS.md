# Journal d'expériences — Benchmark A

Ce fichier consigne chaque exécution significative : **commit hash**, **commande exacte**, **scores**, **artefacts**.

## Entrées

- commit: e48b463
  date: 2025-09-22
  data: data/rfa.csv
  cmd: |
    python -m benchmarks.rfa_ntel --input data/rfa.csv --time-window 24H --ntel-mode R --plot-mean-rt --save-features out/rfa_features_24h.csv
  scores:
    Features_R:  AUC=0.762  ACC=0.723
    Baselines:   AUC=0.989  ACC=0.964
    Combine:     AUC=0.992  ACC=0.968
  artifacts:
    features: out/rfa_features_24h.csv
    plot:     out/mean_rt.png

- commit: e48b463
  date: 2025-09-22
  data: data/rfa.csv
  cmd: |
    python -m benchmarks.rfa_ntel --input data/rfa.csv --msg-window 20 --ntel-mode cos2 --signed --plot-mean-rt --save-features out/rfa_features_m20.csv
  scores:
    Features_R:  AUC=0.786  ACC=0.755
    Baselines:   AUC=0.989  ACC=0.961
    Combine:     AUC=0.990  ACC=0.962
  artifacts:
    features: out/rfa_features_m20.csv
    plot:     out/mean_rt.png

- commit: e48b463
  date: 2025-09-22
  data: data/rfa.csv
  cmd: |
    python -m benchmarks.rfa_ntel --input data/rfa.csv --msg-window 20 --ntel-mode cos2 --signed --early-frac 0.6 --support-band 0.55 0.75 --plot-mean-rt --save-features out/rfa_features_m20_early.csv
  scores:
    Features_R:  AUC=...  ACC=...
    Baselines:   AUC=...  ACC=...
    Combine:     AUC=...  ACC=...
  artifacts:
    features: out/rfa_features_m20_early.csv
    plot:     out/mean_rt.png
