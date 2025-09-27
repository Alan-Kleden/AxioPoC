param(
  [switch]$DoIt,                    # exécute réellement (sinon dry-run)
  [string]$LocalData = "C:\AxioPoC_Data\data",
  [string]$OneDriveData = "$($env:OneDrive)\00_Projets\AxioPoC\data",
  [int]$MinSizeMB = 5               # ignorer les petits fichiers (<5MB)
)

function To-MB($bytes){ [Math]::Round($bytes/1MB,2) }

# 0) contrôles de base
if (-not (Test-Path $LocalData))   { throw "Introuvable: $LocalData" }
if (-not (Test-Path $OneDriveData)){ throw "Introuvable: $OneDriveData" }

Write-Host "Local  (référence): $LocalData"
Write-Host "OneDrive (cible)  : $OneDriveData"
Write-Host "Seuil taille      : $MinSizeMB MB"
Write-Host "Mode              : " -NoNewline
if ($DoIt) {
  Write-Host "EXECUTION (envoi Corbeille)" -ForegroundColor Yellow
} else {
  Write-Host "DRY-RUN (aucune suppression)" -ForegroundColor Cyan
}
Write-Host ""

# 1) Indexer les fichiers locaux (référence) par chemin relatif + taille
$localIndex = @{}
Get-ChildItem -LiteralPath $LocalData -Recurse -File | ForEach-Object {
  $rel = $_.FullName.Substring($LocalData.Length).TrimStart('\')
  $localIndex[$rel] = $_.Length
}

# 2) Chercher, côté OneDrive, les fichiers "doublons" (même chemin relatif) et assez gros
$candidates = @()
Get-ChildItem -LiteralPath $OneDriveData -Recurse -File | ForEach-Object {
  $rel = $_.FullName.Substring($OneDriveData.Length).TrimStart('\')
  if ($localIndex.ContainsKey($rel) -and $_.Length -ge ($MinSizeMB * 1MB)) {
    $sizeMB     = To-MB $_.Length
    $localSizeMB= To-MB $localIndex[$rel]
    $sameSize   = ($_.Length -eq $localIndex[$rel])
    $candidates += [pscustomobject]@{
      RelativePath = $rel
      OneDrivePath = $_.FullName
      OneDriveMB   = $sizeMB
      LocalMB      = $localSizeMB
      SameSize     = $sameSize
    }
  }
}

if (-not $candidates) {
  Write-Host "Aucun doublon >= $MinSizeMB MB détecté entre OneDrive et $LocalData." -ForegroundColor Green
  return
}

# 3) Affichage synthétique
$candidates | Sort-Object -Property OneDriveMB -Descending |
  Select-Object -First 50 RelativePath, OneDriveMB, LocalMB, SameSize |
  Format-Table -AutoSize

$totalMB = ($candidates | Measure-Object OneDriveMB -Sum).Sum
Write-Host ""
Write-Host ("Candidats: {0} fichiers, total ≈ {1} MB" -f $candidates.Count, [math]::Round($totalMB,2))

# 4) Exécution: envoi CORBEILLE (Microsoft.VisualBasic) ou DRY-RUN
if ($DoIt) {
  Add-Type -AssemblyName Microsoft.VisualBasic
  $deleted = 0
  foreach ($c in $candidates) {
    try {
      [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile(
        $c.OneDrivePath,
        'OnlyErrorDialogs',
        'SendToRecycleBin'
      )
      $deleted += 1
      Write-Host "[OK] Corbeille -> $($c.RelativePath) ($($c.OneDriveMB) MB)"
    }
    catch {
      Write-Warning "[ERR] $($c.OneDrivePath) : $($_.Exception.Message)"
    }
  }
  Write-Host ("Terminé: {0}/{1} fichiers envoyés à la Corbeille (~{2} MB)." -f $deleted, $candidates.Count, [math]::Round($totalMB,2)) -ForegroundColor Yellow

  # 5) Option: nettoyer les dossiers vides restants dans OneDrive
  Get-ChildItem -LiteralPath $OneDriveData -Recurse -Directory |
    Sort-Object FullName -Descending | ForEach-Object {
      if (-not (Get-ChildItem -LiteralPath $_.FullName -Force | Where-Object { $_.Name -ne '.gitkeep' })) {
        try { Remove-Item -LiteralPath $_.FullName -Force } catch {}
      }
    }
} else {
  Write-Host ""
  Write-Host "DRY-RUN : rien supprimé. Ajoute -DoIt pour envoyer ces fichiers à la Corbeille." -ForegroundColor Cyan
}
