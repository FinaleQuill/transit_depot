#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$Destination = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Keep this script ASCII-compatible so Windows PowerShell 5.1 can read it
# without depending on the console code page or a UTF-8 BOM.
Add-Type -AssemblyName System.IO.Compression

$projectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = [System.IO.Path]::Combine($projectRoot, "cloud_code.zip")
} elseif (-not [System.IO.Path]::IsPathRooted($Destination)) {
    $Destination = [System.IO.Path]::Combine($projectRoot, $Destination)
}
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
if ([System.IO.Path]::GetExtension($destinationPath) -ine ".zip") {
    throw "Destination must have a .zip extension."
}
if (Test-Path -LiteralPath $destinationPath) {
    throw "Destination already exists; refusing to overwrite: $destinationPath"
}

$separator = [System.IO.Path]::DirectorySeparatorChar
$rootPrefix = $projectRoot.TrimEnd([char[]]@('\', '/')) + $separator
# Original data, caches, accepted manifests, and historical runs remain read-only.
foreach ($protectedName in @(
    "cxr_first", "cache", "data", ".venv", ".venv-competition",
    "runs", "archive", "artifacts"
)) {
    $protectedPath = [System.IO.Path]::Combine($projectRoot, $protectedName)
    $protectedPrefix = $protectedPath.TrimEnd([char[]]@('\', '/')) + $separator
    if ($destinationPath.Equals($protectedPath, [System.StringComparison]::OrdinalIgnoreCase) -or
        $destinationPath.StartsWith($protectedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Destination must be outside protected project directories: $protectedName"
    }
}

# Exact file allowlist: no recursive directory enumeration and no images/weights.
$relativeFiles = @(
    "dataset.py",
    "dataset_highres.py",
    "model.py",
    "model_highres.py",
    "train.py",
    "train_highres.py",
    "inference.py",
    "runtime_compat.py",
    "requirements.txt",
    "run_cloud.sh",
    "pack_cloud.ps1",
    "README.md",
    "AGENTS.md",
    "docs/CLOUD_MIGRATION.md",
    "data/clean_v1_reviewed/train_clean.csv",
    "data/clean_v1_reviewed/valid_manifest.csv",
    "data/clean_v1_reviewed/test_manifest.csv"
)

# Validate every source before creating the archive. Reparse points in the
# allowlisted paths are rejected instead of following links into other trees.
$sourceFiles = @(
    foreach ($relativeName in $relativeFiles) {
        $nativeName = $relativeName.Replace([char]'/', $separator)
        $sourcePath = [System.IO.Path]::GetFullPath(
            [System.IO.Path]::Combine($projectRoot, $nativeName)
        )
        if (-not $sourcePath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Source escaped the project directory: $relativeName"
        }
        if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
            throw "Required allowlisted file is missing: $relativeName"
        }
        $sourceItem = Get-Item -LiteralPath $sourcePath -Force
        $cursor = $sourceItem
        while (-not $cursor.FullName.Equals($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            if (($cursor.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Linked allowlisted files/directories are not supported: $relativeName"
            }
            if ($cursor -is [System.IO.FileInfo]) {
                $cursor = $cursor.Directory
            } else {
                $cursor = $cursor.Parent
            }
            if ($null -eq $cursor) {
                throw "Could not resolve the source parent: $relativeName"
            }
        }
        [pscustomobject]@{ EntryName = $relativeName; SourcePath = $sourcePath }
    }
)

$destinationParent = [System.IO.Path]::GetDirectoryName($destinationPath)
[void][System.IO.Directory]::CreateDirectory($destinationParent)
$utf8 = [System.Text.UTF8Encoding]::new($false, $true)
$created = $false
$completed = $false
try {
    # CreateNew makes refusal to overwrite atomic, including concurrent invocations.
    $outputStream = [System.IO.FileStream]::new(
        $destinationPath, [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write, [System.IO.FileShare]::None
    )
    $created = $true
    try {
        $archive = [System.IO.Compression.ZipArchive]::new(
            $outputStream, [System.IO.Compression.ZipArchiveMode]::Create, $true, $utf8
        )
        try {
            foreach ($source in $sourceFiles) {
                $entry = $archive.CreateEntry(
                    $source.EntryName, [System.IO.Compression.CompressionLevel]::Optimal
                )
                $entryStream = $entry.Open()
                try {
                    if ($source.EntryName -ceq "run_cloud.sh") {
                        # Normalize only archive-entry bytes, never the local source.
                        $scriptText = [System.IO.File]::ReadAllText($source.SourcePath, $utf8)
                        $scriptText = $scriptText.TrimStart([char]0xFEFF)
                        $scriptText = $scriptText.Replace("`r`n", "`n").Replace("`r", "`n")
                        $scriptBytes = $utf8.GetBytes($scriptText)
                        $entryStream.Write($scriptBytes, 0, $scriptBytes.Length)
                    } else {
                        $inputStream = [System.IO.FileStream]::new(
                            $source.SourcePath, [System.IO.FileMode]::Open,
                            [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read
                        )
                        try {
                            $inputStream.CopyTo($entryStream)
                        } finally {
                            $inputStream.Dispose()
                        }
                    }
                } finally {
                    $entryStream.Dispose()
                }
            }
        } finally {
            $archive.Dispose()
        }
    } finally {
        $outputStream.Dispose()
    }
    $completed = $true
} finally {
    if ($created -and -not $completed -and (Test-Path -LiteralPath $destinationPath -PathType Leaf)) {
        # Delete only the incomplete archive created by this invocation.
        Remove-Item -LiteralPath $destinationPath -Force
    }
}

$archiveHash = (Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host ("Created: {0}" -f $destinationPath)
Write-Host ("Files: {0}; original images, caches, environments, runs, and weights excluded." -f $sourceFiles.Count)
Write-Host ("SHA256: {0}" -f $archiveHash)
Write-Host "Upload this ZIP and upload original image data separately. On Linux run: bash run_cloud.sh"
