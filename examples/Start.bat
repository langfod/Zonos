@echo off
setlocal

REM add "-server=<address>" and/or "-port=<port>" arguments to customize server settings
REM add "-server <address>" and/or "-port <port>" arguments to customize server settings
REM - server: The server address to bind to (default: 0.0.0.0)
REM - port: The port to listen on (default: 7860)
REM - device: The device to run the model on (e.g., "cuda:0", "cuda:1"):: Initialize default values

set "server=0.0.0.0"
set "port=7860"
set "device=cuda:0"
:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="-server" (
    set "server=%~2"
    shift
) else if /i "%~1"=="-port" (
    set "port=%~2"
    shift
) else if /i "%~1"=="-device" (
    set "device=%~2"
    shift
)
shift
goto parse_args

:end_parse

call powershell -ExecutionPolicy Bypass -File Start_Zonos.ps1 -server %server% -port %port% -device %device%
