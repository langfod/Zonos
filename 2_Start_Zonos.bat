@echo off


echo Starting Zonos:

ipconfig | find /i "IPv4"

call ".env_win/scripts/activate.bat"

python gradio_interface.py

call ".env_win\Scripts\deactivate.bat"