#requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# === EDITABLES ===
$Seeds    = 0..9
$TestSize = 0.2
$MinMsgs  = 5
$Score    = 'cos'   # 'cos'|'dot'
$Agg      = 'mean'  # 'mean'|'max'

# === PATHS ===
$PY      = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { "python" }
$SCRIPT  = "C:\AxioPoC\scripts\eval_thread_holdout.py"
$EMB     = "H:\AxioPoC\artifacts_xfer\afd_msg_emb_PARTIAL.parquet"
$AXIS    = "C:\AxioPoC\artifacts_xfer\axis_afd.npy"
$LABELS  = "C:\AxioPoC\data\wikipedia\afd\afd_eval.csv"
$TEXTS   = "C:\AxioPoC\artifacts_xfer\afd_messages_filtered.csv"
$ROOT    = "C:\AxioPoC\REPORTS\holdout_grid_${Score}_${Agg}"

@($PY,$SCRIPT,$EMB,$AXIS,$LABELS,$TEXTS) | % { if (-not (Test-Path $_)) { throw "Missing path: $_" } }
New-Item -ItemType Directory -Force -Path $ROOT | Out-Null

$rows = New-Object System.Collections.Generic.List[object]

foreach ($s in $Seeds) {
  $OUTDIR = Join-Path $ROOT ("seed_{0}" -f $s)
  New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null

  & $PY $SCRIPT `
    --emb    $EMB `
    --axis   $AXIS `
    --labels $LABELS `
    --texts  $TEXTS `
    --min-msgs $MinMsgs `
    --score $Score `
    --agg   $Agg `
    --test-size $TestSize `
    --seed  $s `
    --perm-iters 0 `
    --outdir $OUTDIR

  $summary = Join-Path $OUTDIR "summary.json"
  if (Test-Path $summary) {
    $j = Get-Content -Raw $summary | ConvertFrom-Json
    $rows.Add([PSCustomObject]@{
      seed      = $s
      auc_train = [double]$j.auc_train
      auc_test  = [double]$j.auc_test
      acc_test  = [double]$j.acc_test
      thr_youden= [double]$j.thr_opt_youden
      n_threads = [int]$j.n_threads_kept
    }) | Out-Null
  }
}

$csv = Join-Path $ROOT "summary_grid.csv"
$rows | Sort-Object seed | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csv

Write-Host ("`n[OK] Résumé -> {0}" -f $csv)

# Stats rapides en PowerShell pur (pas de Python ici)
if ($rows.Count -gt 0) {
  $aucs = $rows | Select-Object -ExpandProperty auc_test
  $mean = ($aucs | Measure-Object -Average).Average
  $min  = ($aucs | Measure-Object -Minimum).Minimum
  $max  = ($aucs | Measure-Object -Maximum).Maximum
  $med  = ($aucs | Sort-Object | Select-Object -Index ([math]::Floor($aucs.Count/2))).auc_test
  Write-Host ("AUC_test  n={0}  mean={1:N4}  median={2:N4}  min={3:N4}  max={4:N4}" -f $aucs.Count, $mean, $med, $min, $max)
}
