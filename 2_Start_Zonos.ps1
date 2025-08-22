<#
PowerShell equivalent of 2_Start_Zonos.bat
Designed to run on Windows 10 (PowerShell 5.1).

Behavior:
- Display banner
- Attempt to run Visual Studio vcvars scripts (checks for presence and exit code)
- Pause so user can inspect messages
- Start the project using the venv python (if present) in a new window with HIGH priority

Notes:
- If the venv python isn't found this script will try the system python in PATH.
- This script uses cmd.exe start /high to set process priority (works on Windows 10).
#>


function Invoke-Batch($batPath, $arguments) {
    if (-not (Test-Path $batPath)) {
        Write-Host "Batch not found: $batPath" -ForegroundColor Yellow
        return 1
    }
    # Use cmd.exe /c to execute the batch and capture its exit code
    $cmd = "`"$batPath`" $arguments"
    Write-Host "Running: $cmd"
    cmd.exe /c $cmd
    return $LASTEXITCODE
}

function Any_Key_Wait {
    param (
        [string]$msg = "Press any key to continue...",
        [int]$wait_sec = 5
    )
    if ([Console]::KeyAvailable) {[Console]::ReadKey($true) }
    $secondsRunning = $wait_sec;
    Write-Host "$msg" -NoNewline
    While ( !([Console]::KeyAvailable) -And ($secondsRunning -gt 0)) {
        Start-Sleep -Seconds 1;
        Write-Host “$secondsRunning..” -NoNewLine; $secondsRunning--
}

}
Clear-Host


# First attempt: vcvarsall.bat x64
$vcvarsall = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$code = Invoke-Batch -batPath $vcvarsall
if ($code -ne 0) {
    Write-Host "$code" -ForegroundColor Red
    Write-Host "See above. If not Environment initialized for: 'x64' MS Build Tools may be missing consider running 1_Install.bat or using winget to (re)install Microsoft.VisualStudio.2022.BuildTools." -ForegroundColor Yellow
}

Write-Host "`nIf there are errors above, read the messages then press Enter to continue." -ForegroundColor Cyan

Any_Key_Wait 

Write-Host "`nAttempting to start Zonos..." -ForegroundColor Green

# Locate python to run the project. Prefer venv python if present.
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvPython = Join-Path $scriptRoot '.venv\Scripts\python.exe'

if (Test-Path $venvPython) {
    $pythonPath = $venvPython
    Write-Host "Using virtualenv python: $pythonPath"
} else {
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $pythonPath = $pyCmd.Source
        Write-Host "Using system python: $pythonPath"
    } else {
        Write-Host "No python executable found. Please create/activate a virtualenv or install Python and ensure it's in PATH." -ForegroundColor Red
        Read-Host -Prompt "Press Enter to exit"
        exit 1
    }
}

# Script to run (relative to repo root)
$scriptToRun = Join-Path $scriptRoot 'Gradio-zonos.py'
if (-not (Test-Path $scriptToRun)) {
    Write-Host "Could not find script: $scriptToRun" -ForegroundColor Red
    Read-Host -Prompt "Press Enter to exit"
    exit 1
}

# Start a new PowerShell window, set the console title, and run the python script inside it.
Write-Host "Starting new PowerShell window to run: $pythonPath $scriptToRun"

# Build the command to run inside the new PowerShell instance. Escape $Host so it's evaluated by the child PowerShell.
# Detect common Launch-VsDevShell.ps1 locations and prepare init command for the child PowerShell window.
$vsLaunchCandidates = @(
    'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\Launch-VsDevShell.ps1',
    'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1',
    'C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1'
)
$vsLaunch = $vsLaunchCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($vsLaunch) {
    Write-Host "Found Visual Studio dev shell script: $vsLaunch" -ForegroundColor Green
    # Call the launch script inside the child PowerShell so the environment variables it sets are applied there.
    # We intentionally avoid passing unknown/strict parameters so the script uses its sensible defaults.
    $vsInitCommand = "& '$vsLaunch' ;"
} else {
    $vsInitCommand = ""
}

# Build the command to run inside the new PowerShell instance. Escape $Host so it's evaluated by the child PowerShell.
$psCommand = "`$Host.UI.RawUI.WindowTitle = 'Zonos'; $vsInitCommand & '$pythonPath' '$scriptToRun'"

# Launch PowerShell in a new window and keep it open (-NoExit) so errors remain visible.
$proc = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoExit','-Command',$psCommand) -WorkingDirectory $scriptRoot -PassThru
try {
    # Set the PowerShell window process priority to High.
    $proc.PriorityClass = 'High'
    Write-Host "Set PowerShell window process priority to High (Id=$($proc.Id))."
} catch {
    Write-Host "Warning: failed to set process priority: $_" -ForegroundColor Yellow
}

Write-Host "`nZonos should start in another window." -ForegroundColor Green
Write-Host "If that window closes immediately, run $scriptToRun to capture errors." -ForegroundColor Yellow
Any_Key_Wait -msg "Otherwise, you may close this window if it does not close itself.`n" -wait_sec 20
