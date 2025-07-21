import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime

import psutil
import torch


def anonymize_path(path):
    """Replace username in paths with <USER>"""
    if not path:
        return path
    # Handle both Unix and Windows paths
    if path.startswith('/home/'):
        parts = path.split('/')
        if len(parts) > 2:
            parts[2] = '<USER>'
            return '/'.join(parts)
    elif path.startswith('/Users/'):
        parts = path.split('/')
        if len(parts) > 2:
            parts[2] = '<USER>'
            return '/'.join(parts)
    elif path.startswith('C:\\Users\\'):
        parts = path.split('\\')
        if len(parts) > 2:
            parts[2] = '<USER>'
            return '\\'.join(parts)
    return path


def generate_troubleshooting_report(in_model_config_file=None):
    """Generate a comprehensive troubleshooting report for AI/LLM deployment issues."""
    # Create a divider for better readability
    divider = "=" * 80

    # Initialize report
    report = []
    report.append(f"{divider}")
    report.append(f"TROUBLESHOOTING REPORT - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Hardware Information
    report.append("HARDWARE INFORMATION")

    # CPU Info
    report.append("\nCPU:")
    report.append(f"  Model: {platform.processor()}")
    try:
        cpu_freq = psutil.cpu_freq()
        report.append(f"  Max Frequency: {cpu_freq.max:.2f} MHz")
        report.append(f"  Cores: Physical: {psutil.cpu_count(logical=False)}, Logical: {psutil.cpu_count(logical=True)}")
    except Exception as e:
        report.append(f"  Could not get CPU frequency info: {str(e)}")

    # RAM Info
    ram = psutil.virtual_memory()
    report.append("\nRAM:")
    report.append(f"  Total: {ram.total / (1024**3):.2f} GB: free: {ram.available / (1024**3):.2f} used: {ram.used / (1024**3):.2f} GB")

    # GPU Info
    report.append("\nGPU:")
    try:
        nvidia_smi = shutil.which('nvidia-smi')
        if nvidia_smi:
            try:
                gpu_info = subprocess.check_output([nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"], encoding='utf-8').strip()
                gpu_name, vram_total = gpu_info.split(',')
                report.append(f"  Model: {gpu_name.strip()}")
                report.append(f"  VRAM: {vram_total.strip()}")

                try:
                    gpu_usage = subprocess.check_output([nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader"], encoding='utf-8').strip()
                    report.append(f"  VRAM Used: {gpu_usage.strip()}")
                except:
                    pass
            except Exception as e:
                report.append(f"  Could not query GPU info with nvidia-smi: {str(e)}")
    except:
        pass

    # If torch is available and has CUDA, get GPU info from torch
    try:
        if torch.cuda.is_available():
            report.append("\nGPU Info from PyTorch:")
            for i in range(torch.cuda.device_count()):
                report.append(f"  Device {i}: {torch.cuda.get_device_name(i)}, VRAM: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
    except:
        pass

    # Disk Space
    report.append("\nDISK:")
    try:
        disk = psutil.disk_usage('/')
        report.append(f"  Total: {disk.total / (1024**3):.2f} GB.  Free: {disk.free / (1024**3):.2f} GB, Used: {disk.used / (1024**3):.2f} GB")
    except Exception as e:
        report.append(f"  Could not get disk info: {str(e)}")

    # 2. Software Information
    report.append(f"\n{divider}")
    report.append("SOFTWARE INFORMATION")

    # OS Info
    report.append("\nOPERATING SYSTEM:")
    report.append(f"  System: {platform.system()}")
    report.append(f"  Release: {platform.release()}")
    report.append(f"  Version: {platform.version()}")
    report.append(f"  Machine: {platform.machine()}")

    # Python Info
    report.append("\nPYTHON:")
    report.append(f"  Version: {platform.python_version()}")
    report.append(f"  Implementation: {platform.python_implementation()}")
    report.append(f"  Executable: {anonymize_path(sys.executable)}")

    # Installed packages
    report.append("\nINSTALLED PACKAGES (pip freeze):")
    try:
        pip_freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], encoding='utf-8')
        report.append(pip_freeze)
    except Exception as e:
        report.append(f"  Could not get pip freeze output: {str(e)}")

    # CUDA Info
    report.append("CUDA INFORMATION:")
    try:
        nvcc_path = shutil.which('nvcc')
        if nvcc_path:
            nvcc_version = subprocess.check_output(['nvcc', '--version'], encoding='utf-8')
            report.append(nvcc_version.strip())
        else:
            report.append("NVCC not found in PATH")
    except Exception as e:
        report.append(f"  Could not get NVCC version: {str(e)}")

    # PyTorch CUDA version if available
    try:
        if 'torch' in sys.modules:
            report.append("\nPYTORCH CUDA:")
            report.append(f"  PyTorch version: {torch.__version__}")
            report.append(f"  CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                report.append(f"  CUDA version: {torch.version.cuda}")
                report.append(f"  Current device: {torch.cuda.current_device()}")
                report.append(f"  Device name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        report.append(f"  Could not get PyTorch CUDA info: {str(e)}")

    # 3. Model Configuration
    if in_model_config_file:
        report.append(f"\n{divider}")
        report.append("MODEL CONFIGURATION")

        try:
            with open(in_model_config_file, 'r') as f:
                config_content = f.read()
            report.append(f"Content of {anonymize_path(in_model_config_file)}:")
            report.append(config_content)
        except Exception as e:
            report.append(f"\nCould not read model config file {anonymize_path(in_model_config_file)}: {str(e)}")

    # 4. Environment Variables
    report.append(f"\n{divider}")
    report.append("RELEVANT ENVIRONMENT VARIABLES")

    relevant_env_vars = [
        'PATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH',
        'PYTHONPATH', 'CONDA_PREFIX', 'VIRTUAL_ENV'
    ]

    for var in relevant_env_vars:
        if var in os.environ:
            # Anonymize paths in environment variables
            if var in ['PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH']:
                paths = os.environ[var].split(os.pathsep)
                anonymized_paths = [anonymize_path(p) for p in paths]
                report.append(f"{var}: {os.pathsep.join(anonymized_paths)}")
            else:
                report.append(f"{var}: {anonymize_path(os.environ[var])}")

    # 5. Additional System Info
    report.append(f"\n{divider}")
    report.append("ADDITIONAL SYSTEM INFORMATION")

    try:
        # Check if running in container
        report.append("\nContainer/Virtualization:")
        if os.path.exists('/.dockerenv'):
            report.append("  Running inside a Docker container")
        elif os.path.exists('/proc/1/cgroup'):
            with open('/proc/1/cgroup', 'r') as f:
                if 'docker' in f.read():
                    report.append("  Running inside a Docker container")
                elif 'kubepods' in f.read():
                    report.append("  Running inside a Kubernetes pod")
        # Check virtualization
        try:
            virt = subprocess.check_output(['systemd-detect-virt'], encoding='utf-8').strip()
            if virt != 'none':
                report.append(f"  Virtualization: {virt}")
        except:
            pass
    except Exception as e:
        report.append(f"  Could not check container/virtualization info: {str(e)}")

    # Final divider
    report.append("END OF REPORT")
    report.append(f"{divider}")

    # Join all report lines
    full_report = '\n'.join(report)
    return full_report
