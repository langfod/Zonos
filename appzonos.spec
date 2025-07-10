# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_delvewheel_libs_directory
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

data, binary, hiddenimport = collect_all('safehttpx')
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('groovy')
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('inflect')
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('kanjize')
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('language_tags')
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('soundfile')
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('dotenv')
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)


data, binary, hiddenimport = collect_all('psutil')
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('zonos',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('gradio',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('gradio_client',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('torch',include_py_files=True)
print(hiddenimport)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('torch.autograd',include_py_files=True)
print(hiddenimport)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('torchaudio',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('numpy')
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('triton',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('triton.runtime',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('triton.runtime.autotuner',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)


data, binary, hiddenimport = collect_all('triton.backends',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('triton-windows',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('triton-windows.runtime',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('triton-windows.runtime.autotuner',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)


data, binary, hiddenimport = collect_all('triton-windows.backends',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('triton.backends.nvidia',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('triton-windows.backends.nvidia',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport  = collect_all('causal_conv1d',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)


data, binary, hiddenimport  = collect_all('zonos.backbone',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport  = collect_all('zonos.backbone._torch',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport  = collect_all('zonos.backbone._mamba_ssm',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)


data, binary, hiddenimport  = collect_all('zonos.config',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport  = collect_all('mamba_ssm.ops.triton',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport  = collect_all('mamba_ssm.ops',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport  = collect_all('mamba_ssm.ops.triton.layer_norm',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport  = collect_all('mamba_ssm.models.mixer_seq_simple',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

### ,include_py_files=True

data, binary, hiddenimport = collect_all('triton.language',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport = collect_all('jit',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)


data, binary, hiddenimport  = collect_all('flash-attn')
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport  = collect_all('mamba_ssm',include_py_files=True)
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)

data, binary, hiddenimport  = collect_all('causal_conv1d')
datas.extend(data)
binaries.extend(binary)
hiddenimports.extend(hiddenimport)


a = Analysis(
    ['appzonos.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['./extra-hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
	module_collection_mode = {
		'zonos': 'pyz+py', 
		'zonos.backbone': 'pyz+py',
		'mamba_ssm': 'pyz+py',
		'mamba_ssm.ops': 'pyz+py',
		'mamba_ssm.ops.triton': 'pyz+py',
		'torch.autograd': 'pyz+py',		
		'torch-windows.autograd': 'pyz+py',		
		'triton.runtime': 'pyz+py',
		'triton.runtime.autotuner': 'pyz+py',
		'mamba_ssm.ops.triton.layer_norm': 'pyz+py',
		'mamba_ssm.ops.triton.layer_norm.py': 'pyz+py',
		'inflect': 'pyz+py',
    },
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='appzonos',
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
    name='appzonos',
)
