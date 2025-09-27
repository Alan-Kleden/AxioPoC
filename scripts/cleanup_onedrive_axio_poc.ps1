param(
  [switch]$DoIt,  # exécute réellement (sinon dry-run)
  [string]$Root = "$($env:OneDrive)\00_Projets\AxioPoC"
)

function To-MB($bytes){ [Math]::Round($bytes/1MB,2) }

if (-not (Test-Path $Root)) { throw "Dossier OneDrive introuvable: $Root" }

Write-Host "Cible OneDrive : $Root"
Write-Host "Mode           : " -NoNewline
if ($DoIt) { Write-Host "EXECUTION (envoi Corbeille)" -ForegroundColor Yellow } else { Write-Host "DRY-RUN" -ForegroundColor Cyan }
Write-Host ""

# 1) Ce qu’on GARDE (whitelist relative au $Root)
$keep = @(
  "scripts",           # code du PoC
  "REPORTS",           # rapports légers
  "README.md", "README.txt", "README", 
  ".gitignore", ".gitattributes",
  "docs"               # (si tu as un dossier docs léger)
)

# 2) Ce qu’on nettoie de façon agressive (dossiers lourds connus)
$purgeDirs = @(
  "data",              # on videra tout sauf .gitkeep
  "artifacts*",        # tous les artifacts (finals, w20, v10…)
  "venv",".venv","env",
  "logs",
  "__pycache__",".ipynb_checkpoints",
  ".cache",".mypy_cache",".pytest_cache"
)

# 3) Lister contenus du root
$allItems = Get-ChildItem -LiteralPath $Root -Force

# -- A) Candidats à suppression : tout ce qui N’EST PAS dans la whitelist ET qui matche purgeDirs
$candidates = @()

# 3.1 dossiers en racine à purger
foreach($p in $purgeDirs){
  $matches = Get-ChildItem -LiteralPath $Root -Directory -Force -Filter $p -ErrorAction SilentlyContinue
  foreach($m in $matches){
    $candidates += $m.FullName
  }
}

# 3.2 fichiers/dossiers non listés dans la whitelist (sauf keep)
foreach($it in $allItems){
  $rel = $it.Name
  if ($keep -notcontains $rel){
    # si c'est un dossier ET pas déjà dans $candidates, on l’ajoute
    if ($it.PSIsContainer -and ($candidates -notcontains $it.FullName)){
      $candidates += $it.FullName
    }
    # si c'est un fichier lourd hors whitelist, on l’ajoute
    if (-not $it.PSIsContainer -and $it.Length -gt 1MB){
      $candidates += $it.FullName
    }
  }
}

$candidates = $candidates | Sort-Object -Unique

# 4) Affichage synthétique
$rows = foreach($path in $candidates){
  if (Test-Path $path -PathType Container){
    $size = (Get-ChildItem -LiteralPath $path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
  } else {
    $size = (Get-Item -LiteralPath $path -ErrorAction SilentlyContinue).Length
  }
  [pscustomobject]@{ Path=$path; MB=To-MB $size }
}

if (-not $rows){
  Write-Host "Rien à supprimer dans $Root." -ForegroundColor Green
  return
}

$rows | Sort-Object MB -Descending | Format-Table -AutoSize
$totMB = [Math]::Round(($rows | Measure-Object MB -Sum).Sum,2)
Write-Host ""
Write-Host ("TOTAL estimé à libérer (avant corbeille web) : {0} MB" -f $totMB) -ForegroundColor Yellow

# 5) Supprimer (Corbeille) ou Dry-run
if ($DoIt){
  Add-Type -AssemblyName Microsoft.VisualBasic
  foreach($path in $candidates){
    try{
      if (Test-Path $path -PathType Container){
        # supprimer récursif dossier -> corbeille
        # on supprime son contenu dossier par dossier
        Get-ChildItem -LiteralPath $path -Recurse -Force | Sort-Object FullName -Descending | ForEach-Object {
          if (-not $_.PSIsContainer){
            [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($_.FullName, 'OnlyErrorDialogs','SendToRecycleBin')
          }
        }
        # si dossier vide, on le retire
        if (-not (Get-ChildItem -LiteralPath $path -Force -Recurse -ErrorAction SilentlyContinue)){
          Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
        }
      } else {
        [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($path, 'OnlyErrorDialogs','SendToRecycleBin')
      }
      Write-Host "[OK] -> Corbeille : $path"
    } catch {
      Write-Warning "[ERR] $path : $($_.Exception.Message)"
    }
  }

  # 6) Cas particulier : vider le dossier data/ mais garder .gitkeep
  $dataPath = Join-Path $Root "data"
  if (Test-Path $dataPath){
    Get-ChildItem -LiteralPath $dataPath -Recurse -Force | Where-Object {
      -not $_.PSIsContainer -and $_.Name -ne ".gitkeep"
    } | ForEach-Object {
      try {
        [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($_.FullName, 'OnlyErrorDialogs','SendToRecycleBin')
        Write-Host "[OK] data -> Corbeille : $($_.FullName)"
      } catch {
        Write-Warning "[ERR] $($_.FullName) : $($_.Exception.Message)"
      }
    }
    # nettoyage dossiers vides restants (hors data racine)
    Get-ChildItem -LiteralPath $dataPath -Recurse -Directory | Sort-Object FullName -Descending | ForEach-Object {
      if (-not (Get-ChildItem -LiteralPath $_.FullName -Force)){
        Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
      }
    }
  }

  Write-Host "`nFAIT. Pense à vider la **Corbeille OneDrive (web)** pour libérer l’espace."
} else {
  Write-Host "`nDRY-RUN : rien supprimé. Ajoute -DoIt pour exécuter."
}
