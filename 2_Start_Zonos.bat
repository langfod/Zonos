@echo off

echo Starting Zonos in new window...
call .venv\scripts\activate.bat
start "Zonos" /high python appzonos.py
call .venv\Scripts\deactivate.bat