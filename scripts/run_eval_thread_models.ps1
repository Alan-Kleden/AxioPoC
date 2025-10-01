#Requires -Version 5.1
$ErrorActionPreference = "Stop"
$py = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { "python" }

& $py C:\AxioPoC\scripts\eval_thread_models.py `
  --holdout-dir "C:\AxioPoC\REPORTS\holdout_threads" `
  --baselines   "C:\AxioPoC\REPORTS\thread_baselines.csv" `
  --iters 500 `
  --out "C:\AxioPoC\REPORTS\model_comp_summary.json"
