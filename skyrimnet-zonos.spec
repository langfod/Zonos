# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules




datas = []
hiddenimports = []

def add_zonos_files(base_path):
    """Helper function to add all zonos Python and JSON files recursively."""
    zonos_path = Path(base_path)
    if not zonos_path.exists():
        return
    
    for root, dirs, files in os.walk(zonos_path):
        for file in files:
            if file.endswith(('.py', '.json')):
                src_file = Path(root) / file
                rel_path = src_file.relative_to(zonos_path)
                dest_path = f"zonos/{str(rel_path).replace(os.sep, '/')}"
                #print(f"Adding zonos file: {src_file} -> {os.path.dirname(dest_path)}")
                datas.append((str(src_file), os.path.dirname(dest_path)))

# Include all zonos files recursively
add_zonos_files("zonos")

# Core application data files with better exclusions
datas += collect_data_files("gradio_client", excludes=[
    "*.md", "*.txt", "*.rst", "test*", "*test*", "example*", "*example*"
])
datas += collect_data_files("gradio", excludes=[
    "*.md", "*.txt", "*.rst", "test*", "*test*", "demo*"
])

# Keep these smaller libraries as-is (minimal benefit to optimize)
datas += collect_data_files("groovy")
datas += collect_data_files("safehttpx")

# Phonemizer dependency chain - include data files
try:
    datas += collect_data_files("phonemizer")
except Exception as e:
    print(f"Warning: Could not collect phonemizer data files: {e}")

try:
    datas += collect_data_files("segments")
except Exception as e:
    print(f"Warning: Could not collect segments data files: {e}")

try:
    datas += collect_data_files("csvw")
except Exception as e:
    print(f"Warning: Could not collect csvw data files: {e}")

try:
    datas += collect_data_files("language_tags")
except Exception as e:
    print(f"Warning: Could not collect language_tags data files: {e}")

# CRITICAL: Include comprehensive spaCy data files
import os

MODEL_SUPPORTED_LANGS = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja"] # Supported languages from XTTS v2 model config.json


#print(f"Total spaCy data files added: {len([d for d in datas if 'spacy' in d[1]])}")

# Include setuptools data files (needed for jaraco.text and other components)
datas += collect_data_files("setuptools", excludes=[
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "docs/*", "*/docs/*"
])


# CRITICAL: PyTorch with enhanced exclusions to prevent bloat
datas += collect_data_files("torch", excludes=[
    "*.cpp", "*.cu", "*.c", "*.h", "*.cuh",
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*",
    # CRITICAL: Exclude massive library files that cause bloat
    "*.lib", "lib/*.lib", "libs/*.lib",
    # Exclude huge CUDA runtime libraries
    "lib/libtorch_cuda.so*", "lib/libtorch_cpu.a", "lib/dnnl.lib",
    # Multi-GPU solvers (not needed for single-GPU TTS)
    "lib/cusolverMg64_11.dll",   # 179 MB - Multi-GPU solvers
    # NOTE: cuDNN and other CUDA DLL exclusions are handled in post-processing
    # because collect_data_files() doesn't reliably exclude binary dependencies
])

datas += collect_data_files("torchaudio", excludes=[
    "*.cpp", "*.cu", "*.c", "*.h", "*.cuh",
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*"
])


datas += collect_data_files("transformers", excludes=[
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*"
])

# CRITICAL: Include Triton backend data files (driver.py and other backend modules)
# Exclude AMD backend entirely - we only need NVIDIA CUDA backend
datas += collect_data_files("triton", excludes=[
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*",
    "backends/amd/*",  # Exclude AMD backend
    "backends/amd"
])

# CRITICAL: Include mamba_ssm, causal_conv1d, and flash_attn data files
try:
    datas += collect_data_files("mamba_ssm", excludes=[
        "test*", "*test*", "tests/*", "*/tests/*",
        "*.md", "*.txt", "*.rst"
    ])
except Exception as e:
    print(f"Warning: Could not collect mamba_ssm data files: {e}")

try:
    datas += collect_data_files("causal_conv1d", excludes=[
        "test*", "*test*", "tests/*", "*/tests/*",
        "*.md", "*.txt", "*.rst"
    ])
except Exception as e:
    print(f"Warning: Could not collect causal_conv1d data files: {e}")

try:
    datas += collect_data_files("flash_attn", excludes=[
        "test*", "*test*", "tests/*", "*/tests/*",
        "*.md", "*.txt", "*.rst"
    ])
except Exception as e:
    print(f"Warning: Could not collect flash_attn data files: {e}")

# CRITICAL: Include Python development headers for Triton JIT compilation
import sys
import sysconfig
from pathlib import Path

# Get Python installation paths
python_base = Path(sys.base_prefix)
python_include = Path(sysconfig.get_path('include'))
python_stdlib = Path(sysconfig.get_path('stdlib'))

# Include Python headers (Python.h and related files)
if python_include.exists():
    for header_file in python_include.rglob('*.h'):
        rel_path = header_file.relative_to(python_include)
        datas.append((str(header_file), f'include/{rel_path.parent}'))
    print(f"Added Python headers from: {python_include}")

# Include Python libs directory (python3X.lib for linking)
python_libs = python_base / 'libs'
if python_libs.exists():
    for lib_file in python_libs.glob('*.lib'):
        datas.append((str(lib_file), 'libs'))
    print(f"Added Python libs from: {python_libs}")

# Include essential distutils files (needed for compilation)
try:
    datas += collect_data_files("distutils", excludes=[
        "test*", "*test*", "tests/*", "*/tests/*",
        "example*", "*example*", "examples/*", "*/examples/*"
    ])
except:
    # distutils might be built-in, try setuptools._distutils
    try:
        datas += collect_data_files("setuptools._distutils", excludes=[
            "test*", "*test*", "tests/*", "*/tests/*"
        ])
    except:
        pass


# Essential PyTorch modules only
hiddenimports += collect_submodules('torch.nn.functional')


# Fix torch._dynamo import issues (keep minimal)
hiddenimports += collect_submodules('torch._dynamo.polyfills')

# Fix transformers import issues (reduce scope)
hiddenimports += collect_submodules('transformers.generation.utils')
hiddenimports += collect_submodules('transformers.utils')

# CRITICAL: Collect submodules for mamba_ssm, causal_conv1d, flash_attn
try:
    hiddenimports += collect_submodules('mamba_ssm')
except Exception as e:
    print(f"Warning: Could not collect mamba_ssm submodules: {e}")

try:
    hiddenimports += collect_submodules('causal_conv1d')
except Exception as e:
    print(f"Warning: Could not collect causal_conv1d submodules: {e}")

try:
    hiddenimports += collect_submodules('flash_attn')
except Exception as e:
    print(f"Warning: Could not collect flash_attn submodules: {e}")

# Application modules
hiddenimports += [
    "zonos"
]




# Add language-specific modules only if needed
if "ja" in MODEL_SUPPORTED_LANGS:
    hiddenimports += ['sudachipy', 'sudachidict_full', 'kanjize']

# Fix specific import errors (minimal set)
hiddenimports += [
    'torch._dynamo.polyfills.fx',
    # Triton - comprehensive imports for JIT and NVIDIA driver
    "triton",
    "triton.compiler",
    "triton.compiler.compiler",
    "triton.tools",
    "triton.language",
    "triton.runtime",
    "triton.runtime.jit",
    "triton.runtime.driver",
    "triton.runtime.autotuner",
    "triton.backends",
    "triton.backends.nvidia",
    "triton.backends.nvidia.driver",
    "triton.backends.nvidia.compiler",
    # Mamba SSM
    "mamba_ssm",
    "mamba_ssm.models",
    "mamba_ssm.models.mixer_seq_simple",
    "mamba_ssm.ops",
    "mamba_ssm.ops.triton",
    "mamba_ssm.ops.triton.layer_norm",
    # Causal Conv1d
    "causal_conv1d",
    # Flash Attention
    "flash_attn",
]

# CRITICAL: Include compilation modules for Triton JIT
hiddenimports += [
    'distutils',
    'distutils.util',
    'distutils.spawn',
    'distutils.version',
    'setuptools._distutils',
    'sysconfig',
    'subprocess',
    'tempfile',
    # Platform-specific compilation support
    'msvcrt',  # Windows-specific
]

# =============================================================================
# EXCLUDE PROBLEMATIC MODULES (from original)
# =============================================================================

excludedimports = [
    'ninja',
    #'torch.utils.cpp_extension',
    # Exclude AMD backend (we only use NVIDIA CUDA)
    'triton.backends.amd', 
]

# =============================================================================
# ANALYSIS (Keep original configuration mostly)
# =============================================================================

a = Analysis(
    ["SkyrimNet-Zonos.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["pyinstaller-hooks"],
    hooksconfig={},
    runtime_hooks=[
        "pyinstaller-hooks/rthook_disable_typeguard.py",
        "pyinstaller-hooks/rthook_setup_cuda_path.py",
        "pyinstaller-hooks/rthook_triton_nvidia_only.py",
    ],
    excludes=excludedimports,
    noarchive=False,
    optimize=1,  # Conservative optimization
    module_collection_mode={ 
        'zonos.backbone': 'py+pyz',
        'gradio': 'py+pyz',
        'torch': 'py+pyz',
        # CRITICAL: torch._dynamo needs source files for introspection
        'torch._dynamo': 'py',
        'torch._dynamo.polyfills': 'py',
        'torch._dynamo.variables': 'py',
        'torch._inductor': 'py',
        'torch.compiler': 'py',
        # CRITICAL: mamba_ssm uses Triton JIT which requires source files
        'mamba_ssm': 'py',
        'mamba_ssm.ops': 'py',
        'mamba_ssm.ops.triton': 'py',
        'mamba_ssm.modules': 'py',
        # CRITICAL: causal_conv1d uses Triton JIT which requires source files
        'causal_conv1d': 'py',
        # Triton itself needs source files for JIT - include all subpackages
        'triton': 'py',
        'triton.runtime': 'py',
        'triton.runtime.jit': 'py',
        'triton.runtime.driver': 'py',
        'triton.runtime.autotuner': 'py',
        'triton.language': 'py',
        'triton.compiler': 'py',
        'triton.backends': 'py',
        'triton.backends.nvidia': 'py',
        'triton.backends.nvidia.driver': 'py',
        # Flash attention may also use Triton JIT
        'flash_attn': 'py',
        'flash_attn.flash_attn_triton': 'py',
    },
    cipher=None,
    upx=True,
)

# =============================================================================
# NVIDIA CUDA DLL EXCLUSION - Remove system-provided CUDA libraries
# =============================================================================
# These DLLs will be loaded from the system CUDA installation
# CUDA_PATH environment variable points to: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
# DLLs are available in: %CUDA_PATH%\bin\

cuda_dlls_to_exclude = [
    # ULTRA-CONSERVATIVE APPROACH: Only exclude the absolutely safe CUDA runtime DLL
    # Testing shows shm.dll loading issues - reducing exclusions to minimal set
    
    # CUDA Runtime (available in system CUDA installation)
    'cudart64_12.dll',                          # 0.6 MB - CUDA Runtime - SAFE TO EXCLUDE
    
    # Temporarily removing other exclusions to debug shm.dll loading issue
    # Will re-add after confirming application starts correctly
    
    # NOTE: Keep these PyTorch-required DLLs that were previously excluded:
    'cublas64_12.dll', # (97.8 MB) - Required by torch_cuda.dll
    'cublaslt64_12.dll', # (638 MB) - Required by torch_cuda.dll
    'cufft64_11.dll', # (274 MB) - Required by torch_cuda.dll
    'cufftw64_11.dll', # (0.2 MB) - FFTW Interface
    'curand64_10.dll', # (75.5 MB) - Random Number Generation
    'cusolver64_11.dll', # (270 MB) - Required by torch_cuda.dll
    'cusparse64_12.dll', # (455.4 MB) - Required by torch_cuda.dll
    'nvrtc64_120_0.dll', # (85.7 MB) - Runtime Compilation
    # - cudnn64_9.dll (0.3 MB) - Required by torch_cuda.dll
    # - cudnn_cnn64_9.dll (4.4 MB) - Core CNN operations
    # - cudnn_ops64_9.dll (120.6 MB) - Core operations
    # - cudnn_engines_runtime_compiled64_9.dll (19.3 MB) - Runtime engines
    # - cudnn_graph64_9.dll (2.3 MB) - Graph operations
]

# Additional post-processing to remove bloat and CUDA system libraries
a.datas = [x for x in a.datas if not any([
    # Remove .lib files (redundant with collect_data_files exclusions, but some may slip through)
    x[0].lower().endswith('.lib') and 'dnnl' in x[0].lower(),
    x[0].lower().endswith('.lib') and any(huge in x[0].lower() for huge in ['cublas', 'cudnn', 'cufft', 'cusolver']),
    
    # CRITICAL: Exclude NVIDIA CUDA system DLLs - users will have these installed
    any(cuda_dll.lower() in x[0].lower() for cuda_dll in cuda_dlls_to_exclude),
])]

# CRITICAL: Remove NVIDIA CUDA DLLs from binaries as well (they get pulled in as binary dependencies)
a.binaries = [x for x in a.binaries if not any([
    # Exclude all NVIDIA CUDA system DLLs - users will have these installed via CUDA toolkit
    any(cuda_dll.lower() in x[0].lower() for cuda_dll in cuda_dlls_to_exclude),
])]

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# =============================================================================
# EXECUTABLE CONFIGURATION (Keep original structure)
# =============================================================================

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Skyrimnet-Zonos',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,  
    upx=False,    # Disable UPX for now to avoid issues
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    optimize=2,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,   # Disable strip
    upx=False,     # Disable UPX compression to avoid massive files
    upx_exclude=[],
    name='Skyrimnet-Zonos',
    optimize=2,
)