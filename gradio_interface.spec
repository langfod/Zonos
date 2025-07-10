# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = []
datas += collect_data_files('gradio_client')
datas += collect_data_files('gradio')
datas += collect_data_files('torch')
datas += collect_data_files('torchaudio')
datas += collect_data_files('numpy')
datas += collect_data_files('flash-attn')
datas += collect_data_files('triton')

datas += collect_data_files('causal_conv1d')
datas += collect_data_files('mamba_ssm')
datas += collect_data_files('inflect ')
datas += collect_data_files('kanjize')

datas += collect_data_files('soundfile')
datas += collect_data_files('dotenv')



a = Analysis(
    ['gradio_interface.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=2,
	    module_collection_mode={
            'gradio': 'py',
			'torch': 'py',
			'torchaudio': 'py',
			'numpy': 'py',
        },
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='gradio_interface',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='gradio_interface',
)
