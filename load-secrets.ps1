# =============================================================================
# PTCV Secrets Loader (PowerShell)
#
# Loads environment variables from .secrets file.
#
# Usage:
#   . .\load-secrets.ps1           # Load secrets into current session
#   . .\load-secrets.ps1 -Persist  # Also set as User environment variables
# =============================================================================

param(
    [switch]$Persist = $false
)

$secretsFile = Join-Path $PSScriptRoot ".secrets"

if (-not (Test-Path $secretsFile)) {
    Write-Host "[ERROR] Secrets file not found: $secretsFile" -ForegroundColor Red
    Write-Host "[INFO] Copy .secrets.example to .secrets and fill in your credentials" -ForegroundColor Yellow
    return
}

Write-Host "[INFO] Loading secrets from: $secretsFile" -ForegroundColor Green

$loaded = 0
Get-Content $secretsFile | ForEach-Object {
    $line = $_.Trim()

    # Skip empty lines and comments
    if ($line -and -not $line.StartsWith("#")) {
        if ($line -match "^([^=]+)=(.*)$") {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()

            # Set in current session
            [Environment]::SetEnvironmentVariable($name, $value, "Process")

            # Optionally persist to user environment
            if ($Persist) {
                [Environment]::SetEnvironmentVariable($name, $value, "User")
            }

            # Show masked value for confirmation
            $masked = if ($value.Length -gt 8) {
                $value.Substring(0, 4) + "****" + $value.Substring($value.Length - 4)
            } else {
                "****"
            }
            Write-Host "  $name = $masked" -ForegroundColor Cyan
            $loaded++
        }
    }
}

Write-Host "[INFO] Loaded $loaded environment variables" -ForegroundColor Green

if ($Persist) {
    Write-Host "[INFO] Variables also persisted to User environment" -ForegroundColor Yellow
}
