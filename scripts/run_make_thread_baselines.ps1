#Requires -Version 5.1
$ErrorActionPreference = "Stop"
$py = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { "python" }

& $py C:\AxioPoC\scripts\make_thread_baselines.py `
  --texts "C:\AxioPoC\artifacts_xfer\afd_messages_filtered.csv" `
  --labels "C:\AxioPoC\data\wikipedia\afd\afd_eval.csv" `
  --min-msgs 5 `
  --out "C:\AxioPoC\REPORTS\thread_baselines.csv"
