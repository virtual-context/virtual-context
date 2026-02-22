$ErrorActionPreference = "Stop"

$PackageName = "virtual-context"

function Write-Info {
    param([string]$Message)
    Write-Host "[virtual-context] $Message"
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Install-WithPipx {
    if (Test-Command pipx) {
        Write-Info "Installing with pipx..."
        pipx install --force $PackageName
        return $true
    }

    if (Test-Command py) {
        Write-Info "pipx not found. Installing pipx with py -m pip --user..."
        py -m pip install --user pipx
        py -m pipx ensurepath | Out-Null
        if (py -m pipx --version) {
            py -m pipx install --force $PackageName
            return $true
        }
    }

    return $false
}

function Install-WithPipUser {
    if (Test-Command py) {
        Write-Info "Falling back to py -m pip --user install..."
        py -m pip install --user --upgrade $PackageName
        return $true
    }
    return $false
}

Write-Info "Installing $PackageName"

if (-not (Install-WithPipx)) {
    if (-not (Install-WithPipUser)) {
        throw "Installation failed: need Python launcher 'py' and pip, or pipx"
    }
}

if (-not (Test-Command virtual-context)) {
    Write-Info "Installed, but 'virtual-context' is not on PATH yet."
    Write-Info "Open a new PowerShell session, then run: virtual-context --help"
    exit 0
}

Write-Info "Install complete"
Write-Info "Next:"
Write-Info "  1) virtual-context onboard --wizard    # guided setup (config + proxy instances + daemon)"
Write-Info "  Or manual setup:"
Write-Info "  1) virtual-context init coding"
Write-Info "  2) virtual-context config validate"
Write-Info "  3) virtual-context proxy --upstream https://api.anthropic.com"
