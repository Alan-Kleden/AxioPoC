# 0) Python du venv si présent
$py = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { "python" }

# 1) Chemins
$SCRIPT  = "C:\AxioPoC\scripts\face_validity_axis.py"
$EMB     = "H:\AxioPoC\artifacts_xfer\afd_msg_emb_PARTIAL.parquet"
$AXIS    = "C:\AxioPoC\artifacts_xfer\axis_afd.npy"              # changer si besoin
$TEXTS   = "G:\Mon Drive\AxioPoC\artifacts_xfer\afd_messages.csv"
$LABELS  = "C:\AxioPoC\data\wikipedia\afd\afd_eval.csv"
$OUTDIR  = "C:\AxioPoC\REPORTS\face_validity"
$TOPK    = 50

# 2) Sanity
@($SCRIPT,$EMB,$AXIS,$TEXTS,$LABELS) | ForEach-Object {
  if (-not (Test-Path $_)) { throw "Missing path: $_" }
}
New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null

# 3) Run avec labeled-only + seuils
& $py $SCRIPT `
  --emb    "$EMB" `
  --axis   "$AXIS" `
  --texts  "$TEXTS" `
  --labels "$LABELS" `
  --topk   $TOPK `
  --labeled-only `
  --test   auto `
  --p-thresh 0.01 `
  --delta-pp 20 `
  --outdir "$OUTDIR"

# 4) Afficher les sorties importantes
Write-Host "`n--- Outputs ---"
$summary = Join-Path $OUTDIR "summary.json"
$topcsv  = Join-Path $OUTDIR ("top{0}_messages.csv" -f $TOPK)
$botcsv  = Join-Path $OUTDIR ("bottom{0}_messages.csv" -f $TOPK)
@($summary,$topcsv,$botcsv) | ForEach-Object { if (Test-Path $_) { Write-Host $_ } }

if (Test-Path $summary) {
  Write-Host "`n--- summary.json ---"
  Get-Content -Raw $summary | ConvertFrom-Json | ConvertTo-Json -Depth 6
}
