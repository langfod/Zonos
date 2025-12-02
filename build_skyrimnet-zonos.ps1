param(
    [switch]$test,
    [switch]$nobuild,
    [switch]$noarchive,
    [switch]$noclean
)

$PACKAGE_NAME = "SkyrimNet_Zonos"
$version = "1.0.5"
if (-not $nobuild -or $noclean) {
    
    if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
        Write-Host "Virtual environment not found. Please set up the virtual environment before building." -ForegroundColor Red
        exit 1
    }
    & .venv\Scripts\Activate.ps1
    if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
        & pip install pyinstaller
    }
    & pip uninstall pydub -y
    & pip uninstall pydub-ng -y
    & pip install pydub-ng
    Write-Host "Starting build process..."
    if ($noclean) {
        Write-Host "Skipping clean step as per -noclean flag."
        & pyinstaller --noconfirm --log-level=WARN skyrimnet-zonos.spec
    }
    else {
        try {
            if (Test-Path "build") {
                Remove-Item -Path "build" -Recurse -Force
            }
            if (Test-Path "dist") {
                Remove-Item -Path "dist" -Recurse -Force
            }
            if (Test-Path "__pycache__") {
                Remove-Item -Path "__pycache__" -Recurse -Force
            }
        } catch {
            Write-Host "Error during cleanup: $_" -ForegroundColor Red
            exit 1
        }
        & pyinstaller --clean --noconfirm --log-level=ERROR skyrimnet-zonos.spec
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed. Exiting."
        exit $LASTEXITCODE
    }
    Deactivate
}

if ($test) {
    Write-Host "Running in test mode: Archive will be created but not deployed."
    if (-not (Test-Path "dist\skyrimnet-zonos\skyrimnet-zonos.exe")) {
        Write-Host "Error: Executable not found. Please build the project first."
        exit 1
    }
    Copy-Item -Path "configmodel.txt" -Destination "dist\SkyrimNet-Zonos\" -Force
    Copy-Item -Path "models" -Destination "dist\SkyrimNet-Zonos\" -Force -Recurse
    Copy-Item -Path "speakers" -Destination "dist\SkyrimNet-Zonos\" -Force -Recurse
    Copy-Item -Path "assets" -Destination "dist\SkyrimNet-Zonos\" -Force -Recurse
    Copy-Item -Path "examples\Start.bat" -Destination "dist\SkyrimNet-Zonos\" -Force -Recurse
    Copy-Item -Path "examples\Start_Zonos.ps1" -Destination "dist\SkyrimNet-Zonos\" -Force -Recurse


    Set-Location -Path ./dist/SkyrimNet-Zonos
    & ./Start.bat -server "localhost" -port 7860
    Set-Location -Path ../..
}
else {
    Write-Host "Running in deployment mode: Archive will be created and deployed."
    if (-not (Test-Path "dist\skyrimnet-zonos\skyrimnet-zonos.exe")) {
        Write-Host "Error: Executable not found. Please build the project first."
        exit 1
    }

    if (Test-Path "archive") {
        Remove-Item -Path "archive" -Recurse -Force
    }
    New-Item -ItemType Directory -Path "archive/$PACKAGE_NAME" -Force
    New-Item -ItemType Directory -Path "archive/$PACKAGE_NAME/assets" -Force

    Get-ChildItem -Path "speakers" -Directory | Copy-Item -Destination "archive/$PACKAGE_NAME/speakers" 

    Copy-Item -Path "configmodel.txt" -Destination "archive/$PACKAGE_NAME\" -Force
    Copy-Item -Path "speakers\en\malebrute.wav" -Destination "archive/$PACKAGE_NAME/speakers/en\" -Force -Recurse
    Copy-Item -Path "speakers\en\malecommoner.wav" -Destination "archive/$PACKAGE_NAME/speakers/en\" -Force -Recurse
    Copy-Item -Path "assets\silence_100ms.wav" -Destination "archive/$PACKAGE_NAME/assets\" -Force -Recurse
    Copy-Item -Path "examples\Start.bat" -Destination "archive/$PACKAGE_NAME\" -Force
    Copy-Item -Path "examples\Start_Zonos.ps1" -Destination "archive/$PACKAGE_NAME\" -Force
    Copy-Item -Path "dist\skyrimnet-zonos\skyrimnet-zonos.exe" -Destination "archive/$PACKAGE_NAME\" -Force
    Copy-Item -Path "dist\skyrimnet-zonos\_internal" -Destination "archive/$PACKAGE_NAME\" -Force -Recurse


    if (-not $noarchive) {
        $archiveName = "$PACKAGE_NAME"

        if ($version) {
            $archiveName += "_$version"
        }
        Write-Host "Creating archive: $archiveName.zip"
        Set-Location -Path ./archive
        & "C:\Program Files\7-Zip\7z.exe" a -t7z "$archiveName.7z" "$PACKAGE_NAME" -mx=9
    }
}
