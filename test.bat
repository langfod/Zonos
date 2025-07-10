xcopy dist_files\* dist\appzonos /e /i
xcopy dist_files_models\* dist\appzonos /e /i

cd dist\appzonos

appzonos.exe

cd ..\..