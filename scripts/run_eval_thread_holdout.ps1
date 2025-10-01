#requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# === EDITABLES (mets juste ces valeurs) ===
$Seed      = 0
$TestSize  = 0.2
$MinMsgs   = 5
$Score     = 'cos'   # 'cos' ou 'dot'
$Agg       = 'mean'  # 'mean' ou 'max'

# === PATHS (adapte si besoin) ===
$PY      = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { "python" }
$SCRIPT  = "C:\AxioPoC\scripts\eval_thread_holdout.py"
$EMB     = "H:\AxioPoC\artifacts_xfer\afd_msg_emb_PARTIAL.parquet"
$AXIS    = "C:\AxioPoC\artifacts_xfer\axis_afd.npy"
$LABELS  = "C:\AxioPoC\data\wikipedia\afd\afd_eval.csv"
$TEXTS   = "C:\AxioPoC\artifacts_xfer\afd_messages_filtered.csv"
$OUTDIR  = "C:\AxioPoC\REPORTS\holdout_seed$Seed"

# Sanity des chemins + dossier out
@($PY,$SCRIPT,$EMB,$AXIS,$LABELS,$TEXTS) | ForEach-Object {
  if (-not (Test-Path $_)) { throw "Missing path: $_" }
}
New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null

# Run
& $PY $SCRIPT `
  --emb    $EMB `
  --axis   $AXIS `
  --labels $LABELS `
  --texts  $TEXTS `
  --min-msgs $MinMsgs `
  --score $Score `
  --agg   $Agg `
  --test-size $TestSize `
  --seed  $Seed `
  --perm-iters 0 `
  --outdir $OUTDIR

Write-Host "`n[OK] Done -> $OUTDIR"
$summary = Join-Path $OUTDIR "summary.json"
if (Test-Path $summary) {
  Write-Host "`n--- summary.json ---"
  Get-Content -Raw $summary | ConvertFrom-Json | ConvertTo-Json -Depth 6
}
