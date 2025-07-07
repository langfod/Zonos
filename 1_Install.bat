@echo off
echo "Please have Python 3.12"
echo "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"

winget install --id=eSpeak-NG.eSpeak-NG  -e --silent --accept-package-agreements --accept-source-agreements


py -3.12 -m venv .env_win
call ".env_win/scripts/activate.bat"

echo "Installing. Please Wait...."

pip --disable-pip-version-check install --no-clean -r requirements.txt

call ".env_win\Scripts\deactivate.bat"

echo(
echo If all worked ok then run:
echo 2_Start_Zonos.bat