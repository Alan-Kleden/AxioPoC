#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ============ PARAMS ============
$Roots   = @("C:\AxioPoC","H:\AxioPoC","G:\Mon Drive\AxioPoC")
$ReportDir = "C:\AxioPoC\REPORTS\final_poc_afD"
$DoMove = $true
$Stamp  = Get-Date -Format "yyyy-MM-dd_HH-mm"
$ArchiveRoot = "H:\AxioPoC\_archive\$Stamp"

# Dossiers à créer
$null = New-Item -ItemType Directory -Force -Path $ReportDir,$ArchiveRoot

function Write-Info($msg){ Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Dry($msg) { Write-Host "[DRY ] $msg" -ForegroundColor DarkYellow }
function Write-Ok($msg)  { Write-Host "[ OK ] $msg"  -ForegroundColor Green }
function Write-Warn($msg){ Write-Host "[WARN] $msg" -ForegroundColor Yellow }

# ============ 0) INVENTAIRE RAPIDE ============
Write-Info "Inventaire rapide des tailles par racine"
$inv = foreach($r in $Roots){
  if(Test-Path $r){
    $bytes = (Get-ChildItem -Path $r -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    [pscustomobject]@{root=$r; gb=[math]::Round(($bytes/1GB),2)}
  }
}
$inv | Format-Table -Auto
$inv | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $ReportDir "disk_usage_roots_$Stamp.csv")

# ============ 1) COPIE DES RAPPORTS FINaux ============
Write-Info "Copie des rapports finaux dans $ReportDir"
$copySets = @(
  "C:\AxioPoC\REPORTS\face_validity\*",
  "C:\AxioPoC\REPORTS\face_validity_by_thread*\*",
  "C:\AxioPoC\REPORTS\temporal_explore\*",
  "C:\AxioPoC\REPORTS\holdout_seed0\*",
  "C:\AxioPoC\REPORTS\holdout_grid_*\*",
  "C:\AxioPoC\REPORTS\axis_stability*.json",
  "C:\AxioPoC\REPORTS\img\*.png"
)
foreach($pat in $copySets){
  if((Get-ChildItem $pat -ErrorAction SilentlyContinue)){
    Copy-Item $pat $ReportDir -Force -ErrorAction SilentlyContinue
  }
}

# ============ 2) LISTE DE FICHIERS A ARCHIVER ============
Write-Info "Prépare la liste des artefacts à archiver"
$ToArchive = @(
  # Embeddings + textes filtrés
  "H:\AxioPoC\artifacts_xfer\afd_msg_emb_PARTIAL.parquet",
  "C:\AxioPoC\artifacts_xfer\afd_messages_filtered.csv",

  # Axes & signatures
  "C:\AxioPoC\artifacts_xfer\axis_afd*.npy",
  "H:\AxioPoC\artifacts_xfer\signature_*.pkl",

  # Features & permutations
  "H:\AxioPoC\artifacts_xfer\features_*.csv",
  "H:\AxioPoC\artifacts_xfer\permSAFE_*.json"
)

# Intermédiaires / lourds à déplacer aussi
$Patterns = @(
  "C:\AxioPoC\artifacts_xfer\*_shard_*.parquet",
  "C:\AxioPoC\artifacts_xfer\*.tmp",
  "C:\AxioPoC\artifacts_xfer\*.log",
  "H:\AxioPoC\artifacts_xfer\*_shard_*.parquet",
  "H:\AxioPoC\artifacts_xfer\*.tmp",
  "H:\AxioPoC\artifacts_xfer\*.log"
)

# Util: move safe (sans écraser)
function Move-Safe {
  param([string]$Path,[string]$DestRoot)
  if(-not (Test-Path $Path)){ return $null }
  $dest = Join-Path $DestRoot (Split-Path $Path -Leaf)
  New-Item -ItemType Directory -Force -Path $DestRoot | Out-Null
  if(Test-Path $dest){
    $base = [System.IO.Path]::GetFileNameWithoutExtension($dest)
    $ext  = [System.IO.Path]::GetExtension($dest)
    $dest = Join-Path $DestRoot ("{0}__{1}{2}" -f $base,$Stamp,$ext)
  }
  if(-not $DoMove){
    Write-Dry ("MOVE  {0}`n   -> {1}" -f $Path,$dest)
  } else {
    Move-Item -LiteralPath $Path -Destination $dest -Force
    Write-Ok  ("MOVE  {0}`n   -> {1}" -f $Path,$dest)
  }
  return $dest
}

# ============ 3) ARCHIVAGE ============
$Manifest = New-Object System.Collections.Generic.List[object]

foreach($item in $ToArchive + $Patterns){
  $matches = Get-ChildItem $item -ErrorAction SilentlyContinue
  foreach($f in $matches){
    $size = $f.Length
    $dest = Move-Safe -Path $f.FullName -DestRoot $ArchiveRoot
    $Manifest.Add([pscustomobject]@{
      when     = (Get-Date).ToString("s")
      action   = "MOVE"
      source   = $f.FullName
      dest     = $dest
      size_b   = $size
      dry_run  = (-not $DoMove)
    }) | Out-Null
  }
}

$ManifestPath = Join-Path $ReportDir ("cleanup_manifest_{0}.csv" -f $Stamp)
$Manifest | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $ManifestPath
Write-Ok  "Manifest -> $ManifestPath"
Write-Ok  ("Archive  -> {0} (dry-run={1})" -f $ArchiveRoot, (-not $DoMove))
if(-not $DoMove){
  Write-Warn "Ceci était une simulation. Mets `$DoMove = `$true puis relance pour exécuter réellement."
}
