"""
System information utilities for monitoring memory, GPU, and disk usage.
"""
import os
import torch
import psutil


def get_gpu_device():
    """Get the appropriate GPU device (CUDA or MPS) or raise error if none available."""
    if torch.cuda.is_available():
        return torch.device(f'cuda:{torch.cuda.current_device()}')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        raise RuntimeError("No GPU device available. Please use a system with CUDA or MPS support.")


def get_free_system_vram_total_free_used(device=None, debug_mode=False):
    """
    Returns GPU VRAM usage in GB.

    Args:
        device: GPU device to check (defaults to current GPU)
        debug_mode: Whether to print debug information

    Returns:
        tuple: (total_gb, free_gb, used_gb)
    """
    total = 0
    used = 0
    free = 0

    if device is None:
        device = get_gpu_device()

    if device.type == 'mps':
        # MPS doesn't provide detailed memory stats, return a best guess
        bytes_total_available = torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()
        free = torch.mps.recommended_max_memory() / (1024 ** 3)
        used = torch.mps.driver_allocated_memory() / (1024 ** 3)
        total = bytes_total_available / (1024 ** 3)

    elif device.type == 'cuda':
        num_devices = torch.cuda.device_count()
        if debug_mode:
            print(f"Found {num_devices} CUDA device(s)")

        total_vram_all = 0.0
        used_vram_all = 0.0
        free_vram_all = 0.0

        for i in range(num_devices):
            torch.cuda.set_device(i)  # Switch to device `i`
            device = torch.device(f'cuda:{i}')

            # Get memory stats for the current device
            memory_stats = torch.cuda.memory_stats(device)
            bytes_active = memory_stats['active_bytes.all.current']
            bytes_reserved = memory_stats['reserved_bytes.all.current']
            bytes_free_cuda, bytes_total_cuda = torch.cuda.mem_get_info(device)

            # Calculate memory components
            bytes_inactive_reserved = bytes_reserved - bytes_active
            bytes_total_available = bytes_free_cuda + bytes_inactive_reserved

            # Convert to GB
            loop_used = bytes_active / (1024 ** 3)
            loop_free = bytes_total_available / (1024 ** 3)
            loop_total = bytes_total_cuda / (1024 ** 3)

            # Accumulate across all devices
            total_vram_all += loop_total
            used_vram_all += loop_used
            free_vram_all += loop_free

            if debug_mode:
                # Print per-device stats
                print(f"\nDevice {i} ({torch.cuda.get_device_name(i)}):")
                print(f"  Total VRAM: {loop_total:.2f} GB")
                print(f"  Used VRAM:  {loop_used:.2f} GB")
                print(f"  Free VRAM:  {loop_free:.2f} GB")

        if debug_mode:
            # Print aggregated stats
            print("\n=== Total Across All Devices ===")
            print(f"Total VRAM: {total_vram_all:.2f} GB")
            print(f"Used VRAM:  {used_vram_all:.2f} GB")
            print(f"Free VRAM:  {free_vram_all:.2f} GB")

        free = free_vram_all
        total = total_vram_all   # This is more accurate than used+free
        used = total - free

    total = round(total, 2)
    free = round(free, 2)
    used = round(used, 2)

    if debug_mode:
        print(f"GPU mem total: {total}, free: {free}, used: {used}")

    return total, free, used


def get_free_system_ram_total_free_used(debug_mode=False):
    """
    Returns system RAM usage in GB.

    Args:
        debug_mode: Whether to print debug information

    Returns:
        tuple: (total_gb, free_gb, used_gb)
    """
    ram = psutil.virtual_memory()
    total = round(ram.total / (1024**3), 2)
    free = round(ram.available / (1024**3), 2)
    used = round(ram.used / (1024**3), 2)

    if debug_mode:
        print(f"RAM total: {total}, free: {free}, used: {used}")

    return total, free, used


def get_free_system_disk_total_free_used(device=None, debug_mode=False):
    """
    Returns disk usage in GB.

    Args:
        device: Disk device to check (unused, kept for compatibility)
        debug_mode: Whether to print debug information

    Returns:
        tuple: (total_gb, free_gb, used_gb)
    """
    total = 0
    used = 0
    free = 0

    try:
        disk = psutil.disk_usage('/')
        total = round(disk.total / (1024**3), 2)
        free = round(disk.free / (1024**3), 2)
        used = round(disk.used / (1024**3), 2)
    except Exception as e:
        print(f"Could not get disk info: {str(e)}")

    if debug_mode:
        print(f"disk mem total: {total}, free: {free}, used: {used}")

    return total, free, used
