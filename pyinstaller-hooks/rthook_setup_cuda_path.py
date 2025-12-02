"""
Runtime hook to ensure CUDA libraries are available in system PATH.
This hook adds the CUDA installation directory to the system PATH so that
CUDA DLLs can be found at runtime.
"""
import os
import sys

def setup_cuda_path():
    """Add CUDA installation to PATH if available."""
    print("=== CUDA Path Setup Hook ===")
    
    cuda_path = os.environ.get('CUDA_PATH')
    print(f"CUDA_PATH environment variable: {cuda_path}")
    
    if cuda_path:
        if 'CUDA_PATH' in os.environ and 'CUDA_HOME' not in os.environ:
            os.environ['CUDA_HOME'] = os.environ['CUDA_PATH']
            print(f"✅ Set CUDA_HOME to: {os.environ['CUDA_HOME']}")
        cuda_bin = os.path.join(cuda_path, 'bin')
        print(f"Expected CUDA bin directory: {cuda_bin}")
        
        if os.path.exists(cuda_bin):
            # Add CUDA bin directory to PATH if not already present
            current_path = os.environ.get('PATH', '')
            if cuda_bin not in current_path:
                os.environ['PATH'] = cuda_bin + os.pathsep + current_path
                print(f"✅ Added CUDA bin directory to PATH: {cuda_bin}")
            else:
                print(f"✅ CUDA bin directory already in PATH: {cuda_bin}")
            
            # Add as DLL directory for Windows
            if os.name == 'nt':
                try:
                    os.add_dll_directory(cuda_bin)
                    print(f"✅ Added CUDA bin as DLL directory: {cuda_bin}")
                except Exception as e:
                    print(f"⚠️ Could not add DLL directory: {e}")
                
            # List some key CUDA DLLs that should be available
            key_dlls = ['cudart64_12.dll', 'cublas64_12.dll', 'nvrtc64_120_0.dll']
            for dll in key_dlls:
                dll_path = os.path.join(cuda_bin, dll)
                if os.path.exists(dll_path):
                    print(f"✅ Found CUDA DLL: {dll}")
                else:
                    print(f"❌ Missing CUDA DLL: {dll}")
        else:
            print(f"❌ CUDA bin directory not found: {cuda_bin}")
    else:
        print("❌ CUDA_PATH environment variable not set.")
        print("Please ensure NVIDIA CUDA toolkit is installed and CUDA_PATH is set.")
        print("Expected: CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x")
    
    # Also add common CUDA locations to PATH as fallback
    fallback_paths = [
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.7\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin",
    ]
    
    current_path = os.environ.get('PATH', '')
    for fallback_path in fallback_paths:
        if os.path.exists(fallback_path) and fallback_path not in current_path:
            os.environ['PATH'] = fallback_path + os.pathsep + current_path
            print(f"✅ Added fallback CUDA path: {fallback_path}")
            # Also add as DLL directory
            if os.name == 'nt':
                try:
                    os.add_dll_directory(fallback_path)
                except:
                    pass
            current_path = os.environ['PATH']
            break
    
    print("=== End CUDA Path Setup ===\n")

# Call setup function when this hook is loaded
setup_cuda_path()