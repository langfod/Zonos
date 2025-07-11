@echo off
echo "Please have Python 3.12"
echo "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
echo(

echo(
echo "Checking for eSpeak NG" 
echo(
winget install --id=eSpeak-NG.eSpeak-NG  -e --silent --accept-package-agreements --accept-source-agreements


echo (
py -3.12 -m venv --clear --upgrade-deps .env_win
call ".env_win/scripts/activate.bat"

echo "Installing. Please Wait...."

pip --disable-pip-version-check install --no-clean -r requirements.txt

 python -c "import gradio_interface; gradio_interface.load_model_if_needed('Zyphra/Zonos-v0.1-hybrid')"
 python -c "import gradio_interface; gradio_interface.load_model_if_needed('Zyphra/Zonos-v0.1-transformer')"
 
call ".env_win\Scripts\deactivate.bat"

echo(
echo If all worked ok then run:
echo 2_Start_Zonos.bat