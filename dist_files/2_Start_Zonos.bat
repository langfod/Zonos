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



call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64  || set "_error=1"
if %errorlevel% NEQ 0 goto error
if "!_error!"=="1" goto error




echo *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#**
echo * IF THERE ARE ERRORS ABOVE THE READ THIS THE EXIT WITH CTRL-C
echo *
echo * x64 MS Build tools not working.
echo * rerun 1_Install.bat
echo *
echo * or
echo * winget uninstall --id Microsoft.VisualStudio.2022.BuildTools
echo * winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools;includeRecommended" --silent --accept-package-agreements --accept-source-agreements
echo *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#**

echo #
echo # Otherwisse just hit any key:
pause

start "SkyrimNet Zonos" /high skyrimnet-zonos.exe



:okay
echo *************************************************************************************
echo *
echo *  SkyrimNet Zonos should start in another window.
echo *
echo *   If the windows closes immediately the run SkyrimNet-Zonos.exe and send the error.
echo *
echo *  Otherwise you may close this window now
echo **************************************************************************************

goto eof

:error
echo *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#**
echo * Something is missing or broken...
echo *
echo * x64 MS Build tools not working.
echo * rerun 1_Install.bat
echo *
echo * or
echo * winget uninstall --id Microsoft.VisualStudio.2022.BuildTools
echo * winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools;includeRecommended" --silent --accept-package-agreements --accept-source-agreements
echo *#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#**

goto eof


:eof
pause
