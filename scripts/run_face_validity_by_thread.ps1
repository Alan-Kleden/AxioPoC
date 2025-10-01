#Requires -Version 5.1
$ErrorActionPreference = "Stop"

# Python (venv si dispo)
$py = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { "python" }

# Paramètres
$SCRIPT = "C:\AxioPoC\scripts\face_validity_axis_by_thread.py"
$EMB    = "H:\AxioPoC\artifacts_xfer\afd_msg_emb_PARTIAL.parquet"
$AXIS   = "C:\AxioPoC\artifacts_xfer\axis_afd.npy"           # ou axis_afd_early50.npy
$TEXTS  = "G:\Mon Drive\AxioPoC\artifacts_xfer\afd_messages.csv"
$LABELS = "C:\AxioPoC\data\wikipedia\afd\afd_eval.csv"
$OUTDIR = "C:\AxioPoC\REPORTS\face_validity_by_thread"

# Création dossier
New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null

# --- Run #1: dot + mean, min_msgs=5 ---
& $py $SCRIPT `
  --emb "$EMB" `
  --axis "$AXIS" `
  --texts "$TEXTS" `
  --labels "$LABELS" `
  --outdir "$OUTDIR" `
  --topk 50 `
  --score dot `
  --agg mean `
  --min_msgs 5

# --- Run #2: cos + mean (optionnel, pour comparer) ---
& $py $SCRIPT `
  --emb "$EMB" `
  --axis "$AXIS" `
  --texts "$TEXTS" `
  --labels "$LABELS" `
  --outdir "$OUTDIR" `
  --topk 50 `
  --score cos `
  --agg mean `
  --min_msgs 5

# Affiche le résumé s’il existe
$sum = Join-Path $OUTDIR "summary_by_thread.json"
if (Test-Path $sum) {
  Get-Content -Raw $sum | ConvertFrom-Json | ConvertTo-Json -Depth 6
}
