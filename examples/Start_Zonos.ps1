<#
PowerShell equivalent of 2_Start_Zonos.bat
Designed to run on Windows 10 (PowerShell 5.1).

Behavior:
- Display banner
- Attempt to run Visual Studio vcvars scripts (checks for presence and exit code)
- Pause so user can inspect messages
- Start the project using the venv python (if present) in a new window with HIGH priority

Parameters:
- server: The server address to bind to (default: 0.0.0.0)
- port: The port to listen on (default: 7860)
- device: The device to run the model on (e.g., "cuda:0", "cuda:1") (optional)

Notes:
- If the venv python isn't found this script will try the system python in PATH.
- This script uses cmd.exe start /high to set process priority (works on Windows 10).
#>

param(
    [string]$server = "0.0.0.0",
    [int]$port = 7860,
    [string]$device = "cuda:0"
)

function Show-Banner {
    $banner = @'
  ad88888ba   88                                 88                      888b      88                       
 d8"     "8b  88                                 ""                      8888b     88                ,d     
 Y8,          88                                                         88 `8b    88                88     
 `Y8aaaaa,    88   ,d8  8b       d8  8b,dPPYba,  88  88,dPYba,,adPYba,   88  `8b   88   ,adPPYba,  MM88MMM  
   `""""""8b,  88 ,a8"   `8b     d8'  88P'   "Y8  88  88P'   "88"    "8a  88   `8b  88  a8P_____88    88     
         `8b  8888[      `8b   d8'   88          88  88      88      88  88    `8b 88  8PP"""""""    88     
 Y8a     a8P  88`"Yba,    `8b,d8'    88          88  88      88      88  88     `8888  "8b,   ,aa    88,    
  "Y88888P"   88   `Y8a     Y88'     88  _____   88  88      88      88  88      `888   `"Ybbd8"'    "Y888  
                            d8'         |__  /___  _ __   ___  ___                                          
                           d8'            / // _ \| '_ \ / _ \/ __|                                         
                                         / /| (_) | | | | (_) \__ \                                         
                                        /____\___/|_| |_|\___/|___/                                         
                                                                                                            
 #########################################################################################################  
 #                                                                                                       #  
 #  Checking for eSpeak and MS Build Tools                                                               #  
 #                                                                                                       #  
 # You may need to accept installation windows                                                           #  
 # and close them when they complete.                                                                    #  
 #                                                                                                       #  
 #########################################################################################################  
'@

    Write-Host $banner
}

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
Show-Banner

# Set up Visual Studio 2022 x64 environment
$currentDirectory = $PWD.Path
& 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\Tools\\Launch-VsDevShell.ps1' -Arch amd64; Set-Location -Path "${currentDirectory}"

Write-Host "`nAttempting to start SkyrimNet Zonos..." -ForegroundColor Green

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$exePath = Join-Path $scriptRoot 'skyrimnet-zonos.exe'

    if (!$exePath) {
        Write-Host "No executable found" -ForegroundColor Red
        Read-Host -Prompt "Press Enter to exit"
        exit 1
    }


$exeArgs = "--server $server --port $port --device $device"

# Start a new PowerShell window, set the console title, and run the python module inside it.
if ($exeArgs) {
    Write-Host "Starting new PowerShell window to run: $exePath $exeArgs"
} else {
    Write-Host "Starting new PowerShell window to run: $exePath (with no arguments)"
}

# Build the command to run inside the new PowerShell instance. Escape $Host so it's evaluated by the child PowerShell.
$psCommand = "`$Host.UI.RawUI.WindowTitle = 'SkyrimNet Zonos'; & '$exePath' $exeArgs"

# Launch PowerShell in a new window and keep it open (-NoExit) so errors remain visible.
$proc = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoExit','-Command',$psCommand) -WorkingDirectory $scriptRoot -PassThru
try {
    # Set the PowerShell window process priority to High.
    $proc.PriorityClass = 'High'
    Write-Host "Set PowerShell window process priority to High (Id=$($proc.Id))."
} catch {
    Write-Host "Warning: failed to set process priority: $_" -ForegroundColor Yellow
}

Write-Host "`nSkyrimNet Zonos should start in another window. Default web server is http://localhost:$port" -ForegroundColor Green
Any_Key_Wait -msg "Otherwise, you may close this window if it does not close itself.`n" -wait_sec 20
