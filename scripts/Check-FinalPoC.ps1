# Check-FinalPoC.ps1
# Validates Final PoC folder contents and lists large files
# Exit codes:
#   0 = OK
#   1 = Final folder missing
#   2 = Required files missing

[CmdletBinding()]
param(
  [string]$FinalDir = "C:\AxioPoC\REPORTS\final_poc_afD",
  [int]$LargeThresholdMB = 20
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Fail($code, $msg) {
  Write-Host "[FAIL] $msg" -ForegroundColor Red
  exit $code
}
function Ok($msg) {
  Write-Host "[OK] $msg" -ForegroundColor Green
}

# 1) Final folder exists?
if (-not (Test-Path -LiteralPath $FinalDir)) {
  Fail 1 "Final folder not found: $FinalDir"
}
Ok "Final folder found: $FinalDir"

# 2) Required files present?
$required = @("mini_rapport_final.md","README.md","file_index.csv")
$missing = @()
foreach ($name in $required) {
  $p = Join-Path $FinalDir $name
  if (-not (Test-Path -LiteralPath $p)) { $missing += $p }
}
if ($missing.Count -gt 0) {
  Write-Host "Missing required files:" -ForegroundColor Yellow
  $missing | ForEach-Object { Write-Host " - $_" -ForegroundColor Yellow }
  Fail 2 "Please add missing files above before committing."
}
Ok "All required files are present: $($required -join ', ')"

# 3) List large files (> threshold)
$thresholdBytes = [int64]$LargeThresholdMB * 1MB
$large = @(
  Get-ChildItem -LiteralPath $FinalDir -Recurse -File |
  Where-Object { $_.Length -gt $thresholdBytes } |
  Select-Object FullName,
                @{n='SizeMB';e={[math]::Round($_.Length / 1MB,2)}},
                LastWriteTime
)

if (@($large).Count -eq 0) {
  Ok "No files larger than $LargeThresholdMB MB in $FinalDir"
} else {
  $csvPath = Join-Path $FinalDir ("large_files_over_{0}MB.csv" -f $LargeThresholdMB)
  $large | Sort-Object SizeMB -Descending |
    Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csvPath
  Write-Host ("[INFO] Large files (> {0} MB):" -f $LargeThresholdMB) -ForegroundColor Cyan
  $large | Sort-Object SizeMB -Descending | Format-Table -AutoSize
  Ok "CSV written: $csvPath"
}

Ok "Final PoC folder is ready for commit."
exit 0
