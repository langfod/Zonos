@echo off

set "tempFile=winget_output.txt"

echo "  ad88888ba   88                                 88                      888b      88                       "
echo " d8"     "8b  88                                 ""                      8888b     88                ,d     "
echo " Y8,          88                                                         88 `8b    88                88     "
echo " `Y8aaaaa,    88   ,d8  8b       d8  8b,dPPYba,  88  88,dPYba,,adPYba,   88  `8b   88   ,adPPYba,  MM88MMM  "
echo "   `"""""8b,  88 ,a8"   `8b     d8'  88P'   "Y8  88  88P'   "88"    "8a  88   `8b  88  a8P_____88    88     "
echo "         `8b  8888[      `8b   d8'   88          88  88      88      88  88    `8b 88  8PP"""""""    88     "
echo " Y8a     a8P  88`"Yba,    `8b,d8'    88          88  88      88      88  88     `8888  "8b,   ,aa    88,    "
echo "  "Y88888P"   88   `Y8a     Y88'     88  _____   88  88      88      88  88      `888   `"Ybbd8"'    "Y888  "
echo "                            d8'         |__  /___  _ __   ___  ___                                          "
echo "                           d8'            / // _ \| '_ \ / _ \/ __|                                         "
echo "                                         / /| (_) | | | | (_) \__ \                                         "
echo "                                        /____\___/|_| |_|\___/|___/                                         "
echo "                                                                                                            "
echo " #########################################################################################################  "
echo " #                                                                                                       #  "
echo " #  Checking for eSpeak and MS Build Tools                                                               #  "
echo " #                                                                                                       #  "
echo " # You may need to accept installation windows                                                           #  "
echo " # and close them when they complete.                                                                    #  "
echo " #                                                                                                       #  "
echo " #########################################################################################################  "

  

echo ###
echo # Checking if eSpeak-NG.eSpeak-NG is installed...
echo ###

winget list --id eSpeak-NG.eSpeak-NG > "%tempFile%"

findstr /i /c:eSpeak-NG.eSpeak-NG "%tempFile%" >nul
if %errorlevel% NEQ 0 (
    echo "eSpeak-NG.eSpeak-NG" is NOT installed.
    winget install --id=eSpeak-NG.eSpeak-NG  -e --silent --accept-package-agreements --accept-source-agreements
)
if %errorlevel% NEQ 0 goto error
del %tempFile%


echo ###
echo # Not trusting the Microsoft.VisualStudio.2022.BuildTools so uninstalling and reinstalling with x64
REM echo # Checking if Microsoft.VisualStudio.2022.BuildTools is installed (should be x64 but cannot check...
echo ###
winget uninstall --id Microsoft.VisualStudio.2022.BuildTools
winget list --id Microsoft.VisualStudio.2022.BuildTools > "%tempFile%"

findstr /i /c:Microsoft.VisualStudio.2022.BuildTools "%tempFile%" >nul
if %errorlevel% NEQ 0 (
    echo Microsoft.VisualStudio.2022.BuildTools is NOT installed.
    winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools;includeRecommended" --silent --accept-package-agreements --accept-source-agreements
)
if %errorlevel% NEQ 0 goto error
del %tempFile%


goto endokay


:error
echo "Something is missing or broken..."
goto eof


:endokay
echo ***************************************************
echo *
echo *   If all succeeded then you can run 2_Start.bat
echo *
echo ***************************************************

:eof
pause
