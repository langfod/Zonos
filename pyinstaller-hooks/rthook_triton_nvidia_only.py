"""
PyInstaller runtime hook to enable Triton JIT compilation in frozen environment.

This sets up the necessary environment for Triton to compile CUDA kernels
at runtime, including Python headers and compilation paths.
"""
import os
import sys
from pathlib import Path

# Detect PyInstaller frozen environment
IS_FROZEN = hasattr(sys, '_MEIPASS')

# Set up Python compilation environment for Triton JIT
if IS_FROZEN:
    meipass = Path(sys._MEIPASS)
    
    # Set Python include path for headers (Python.h)
    python_include = meipass / 'include'
    if python_include.exists():
        os.environ['PYTHON_INCLUDE'] = str(python_include)
    
    # Set Python libs path for linking
    python_libs = meipass / 'libs'
    if python_libs.exists():
        os.environ['PYTHON_LIBS'] = str(python_libs)
    
    # Set TRITON_CACHE_DIR to writable temp directory
    import tempfile
    triton_cache = Path(tempfile.gettempdir()) / 'triton_cache'
    triton_cache.mkdir(exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(triton_cache)
    
    # Enable Triton JIT compilation (remove interpret mode)
    os.environ.pop('TRITON_INTERPRET', None)
    
    print(f"Triton JIT enabled with cache: {triton_cache}")
    if python_include.exists():
        print(f"Python headers available: {python_include}")
    if python_libs.exists():
        print(f"Python libs available: {python_libs}")

# Restrict Triton to NVIDIA backend only
os.environ['TRITON_BACKENDS'] = 'nvidia'


def _setup_triton_nvidia_backend():
    """
    Manually set up the Triton NVIDIA backend for PyInstaller frozen environment.
    
    In frozen apps, the entry_points-based backend discovery may fail.
    This function manually imports and registers the NVIDIA backend.
    """
    try:
        print("=== Setting up Triton NVIDIA Backend ===")
        
        # Import triton components
        import triton
        from triton import backends as triton_backends
        from triton.runtime import driver as runtime_driver
        
        # Import NVIDIA backend components directly
        from triton.backends.nvidia.driver import CudaDriver
        from triton.backends.nvidia.compiler import CUDABackend
        
        # Check if backends are already discovered
        if hasattr(triton_backends, 'backends') and triton_backends.backends:
            if 'nvidia' in triton_backends.backends:
                print(f"✅ NVIDIA backend already registered: {triton_backends.backends['nvidia']}")
            else:
                print(f"⚠️ Backends found but no NVIDIA: {list(triton_backends.backends.keys())}")
        else:
            print("⚠️ No backends discovered, manually registering NVIDIA...")
            
            # Manually create and register the backend
            from triton.backends import Backend
            nvidia_backend = Backend(compiler=CUDABackend, driver=CudaDriver)
            
            if not hasattr(triton_backends, 'backends') or triton_backends.backends is None:
                triton_backends.backends = {}
            
            triton_backends.backends['nvidia'] = nvidia_backend
            print(f"✅ Manually registered NVIDIA backend: {nvidia_backend}")
        
        # Ensure the active driver is set
        if hasattr(runtime_driver, 'active'):
            try:
                # Try to access the active driver to trigger initialization
                active = runtime_driver.active
                if hasattr(active, '_obj') and active._obj is None:
                    print("⚠️ Active driver not initialized, setting manually...")
                    runtime_driver.set_active(CudaDriver())
                    print("✅ Manually set active driver to CudaDriver")
                else:
                    print(f"✅ Active driver: {active}")
            except Exception as e:
                print(f"⚠️ Could not check active driver: {e}")
                # Try to set it anyway
                try:
                    runtime_driver.set_active(CudaDriver())
                    print("✅ Set active driver to CudaDriver")
                except Exception as e2:
                    print(f"❌ Could not set active driver: {e2}")
        
        print("=== End Triton NVIDIA Backend Setup ===\n")
        
    except ImportError as e:
        print(f"⚠️ Could not import Triton components: {e}")
    except Exception as e:
        print(f"⚠️ Error during Triton NVIDIA backend setup: {e}")
        import traceback
        traceback.print_exc()


# Apply setup when this hook runs
_setup_triton_nvidia_backend()
