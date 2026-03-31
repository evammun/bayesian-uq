# ============================================================================
# Fetch results from CSC Mahti (PowerShell)
# ============================================================================
# Usage:
#   One-shot:  .\scripts\mahti\fetch_results.ps1
#   Polling:   .\scripts\mahti\fetch_results.ps1 -Poll -IntervalSec 300
# ============================================================================

param(
    [string]$SshHost = "evamar@mahti.csc.fi",
    [switch]$Poll,
    [int]$IntervalSec = 300
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
$LocalResults = Join-Path $ProjectRoot "results"
$RemoteDir = "/scratch/project_2018384/bayesian-uq/results"

if (-not (Test-Path $LocalResults)) {
    New-Item -ItemType Directory -Path $LocalResults | Out-Null
}

function Get-LocalQuestionCount($filePath) {
    if (-not (Test-Path $filePath)) { return 0 }
    $content = [System.IO.File]::ReadAllText($filePath)
    return ([regex]::Matches($content, '"question_id"')).Count
}

function Do-Fetch {
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[$ts] Checking Mahti for new/updated results..."

    try {
        $remoteList = ssh $SshHost "ls -l $RemoteDir/*.json 2>/dev/null | awk '{print `$NF, `$5}'" 2>$null
    } catch {
        Write-Host "[$ts] SSH failed"
        return
    }

    if (-not $remoteList) {
        Write-Host "[$ts] No result files on remote."
        return
    }

    $toFetch = @()
    $skipped = 0

    foreach ($line in $remoteList -split "`n") {
        $parts = $line.Trim() -split '\s+'
        if ($parts.Count -lt 2) { continue }
        $remotePath = $parts[0]
        $remoteSize = [long]$parts[1]
        $fileName = Split-Path -Leaf $remotePath
        $localPath = Join-Path $LocalResults $fileName

        if (Test-Path $localPath) {
            $localSize = (Get-Item $localPath).Length
            if ($localSize -eq $remoteSize) {
                $skipped++
                continue
            }
        }
        $toFetch += $remotePath
    }

    if ($toFetch.Count -eq 0) {
        Write-Host "[$ts] All $skipped files up to date."
    } else {
        Write-Host "[$ts] Fetching $($toFetch.Count) file(s) ($skipped unchanged)..."
        foreach ($remote in $toFetch) {
            $fileName = Split-Path -Leaf $remote
            Write-Host "  -> $fileName"
            scp "${SshHost}:${remote}" "$LocalResults/" 2>$null
        }
    }

    $files = Get-ChildItem "$LocalResults\*.json" -ErrorAction SilentlyContinue
    Write-Host "[$ts] Local: $($files.Count) result files"
    foreach ($f in $files | Sort-Object Name) {
        $n = Get-LocalQuestionCount $f.FullName
        $name = $f.BaseName -replace '_\d{8}_\d{6}$', ''
        $status = if ($n -ge 4609) { "DONE" } else { "$n/4609" }
        $fetched = if ($toFetch | Where-Object { (Split-Path -Leaf $_) -eq $f.Name }) { " *" } else { "" }
        Write-Host ("  {0,-55} {1}{2}" -f $name, $status, $fetched)
    }
    Write-Host ""
}

if ($Poll) {
    Write-Host "============================================"
    Write-Host "  Polling Mahti results every ${IntervalSec}s"
    Write-Host "  Press Ctrl+C to stop"
    Write-Host "============================================"
    Write-Host ""
    while ($true) {
        Do-Fetch
        Write-Host "--- Next fetch in ${IntervalSec}s ---"
        Write-Host ""
        Start-Sleep -Seconds $IntervalSec
    }
} else {
    Do-Fetch
}
