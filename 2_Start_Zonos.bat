@echo off

echo Starting Zonos:

call ".env_win\Scripts\deactivate.bat"

call ".venv/scripts/activate.bat"

start "Zonos" /high python appzonos.py

call ".venv\Scripts\deactivate.bat"