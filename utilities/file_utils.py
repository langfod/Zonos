"""
File system utilities for path checking and validation.
"""
import os
from pathlib import Path
from typing import List, Tuple, Dict, Set


def count_existing_paths(paths: List[str]) -> Tuple[str, bool, bool, List[str]]:
    """
    Checks if each path in the list exists.
    Returns:
        - summary (str): Summary of found/missing count
        - all_found (bool): True if all paths were found
        - none_found (bool): True if no paths were found
        - details (list of str): List with "[found]" or "[not found]" per path
    """
    total = len(paths)
    if total == 0:
        return "No paths provided.", False, True, []
    found_count = 0
    details = []
    for path in paths:
        if os.path.exists(path):
            found_count += 1
            details.append(f"[!FOUND!]: {path}")
        else:
            details.append(f"[MISSING]: {path}")
    missing_count = total - found_count
    all_found = (missing_count == 0)
    none_found = (found_count == 0)
    summary = f"Found {found_count}, missing {missing_count}, out of {total} paths."
    return summary, all_found, none_found, details


def remove_suffix(text: str, suffix: str) -> str:
    """Remove suffix from text if it exists."""
    if text.endswith(suffix):
        return text[:-len(suffix)]
    return text


def get_hf_model_cache_dirname(model_id: str) -> str:
    """
    Returns the HF cache directory name for a given model.
    """
    base = "models--"
    return base + model_id.replace('/', '--')


def check_do_all_files_exist(
    dotenv_needed_models: Set[str],
    dotenv_loaded_paths: Dict[str, str],
    dotenv_loaded_models: Dict[str, str],
    dotenv_loaded_models_values: Dict[str, str],
    in_files_to_check_in_paths: List[Dict[str, str]] = None,
    silent: bool = False,
    debug_mode: bool = False,
    app_name: str = "FILE_CHECK"
) -> bool:
    """
    Check if all required model files and paths exist.

    Returns:
        bool: True if all files exist, False otherwise
    """
    test_models_hf = []
    test_models_dir = []
    test_paths_dir = []

    retval_models_exist = True
    retval_paths_exist = True

    if in_files_to_check_in_paths is None:
        in_files_to_check_in_paths = []

    # add model paths as path and as hf cache path
    for currmodel in dotenv_needed_models:
        test_models_hf.append(f"{dotenv_loaded_paths['HF_HOME']}{os.sep}hub{os.sep}{get_hf_model_cache_dirname(currmodel)}{os.sep}snapshots")
        test_models_dir.append(f"{dotenv_loaded_models[currmodel]}")

    # add needed dirs as path
    for curr_path in dotenv_loaded_paths:
        test_paths_dir.append(f"{dotenv_loaded_paths[curr_path]}")

    if debug_mode:
        print(f"test path hf: {test_models_hf}")
        print(f"test path dirs: {test_models_dir}")

    if not silent:
        print(f"{app_name}: checking model accessibility")
    summary_hf, all_exist_hf, none_exist_hf, path_details_hf = count_existing_paths(test_models_hf)

    if not silent:
        print(f"\n-Searching Group1: Model HF_HOME----------------------------------------------")
        for line in path_details_hf:
            print_line = remove_suffix(line, "snapshots")
            print(print_line)

    summary_dir, all_exist_dir, none_exist_dir, path_details_dir = count_existing_paths(test_models_dir)
    if not silent:
        print("-Searching Group2: Manual Model Directories-----------------------------------")
        for line in path_details_dir:
            print_line = remove_suffix(line, "model_index.json")
            print_line = remove_suffix(print_line, "config.json")
            print(print_line)

    summary_path, all_exist_path, none_exist_path, path_details_path = count_existing_paths(test_paths_dir)
    if not silent:
        print("-Searching Group3: Needed Directories-----------------------------------------")
        for line in path_details_path:
            print(line)

    if not silent:
        print("-checking explicit Files---------------------------------------------------")

    for mapping in in_files_to_check_in_paths:
        for env_var, relative_path in mapping.items():
            if dotenv_loaded_paths and env_var in dotenv_loaded_paths:
                base_path = dotenv_loaded_paths[env_var]
                full_path = Path(base_path) / relative_path.strip(os.sep)
                if full_path.exists():
                    if not silent:
                        print(f"[!FOUND!]: {full_path}")
                else:
                    if not silent:
                        print(f"[!MISSING!]: {full_path}")
                    retval_paths_exist = False
    if not silent:
        print("")

    # we show the dir values to the user
    if not silent:
        if all_exist_dir == False:
            print("-Values in config (resolved to your OS)---------------------------------------")
            for key in dotenv_loaded_models_values:
                print(f"{key}: {os.path.abspath(dotenv_loaded_models_values[key])}")
        if all_exist_path == False:
            for key in dotenv_loaded_paths:
                print(f"{key}: {os.path.abspath(dotenv_loaded_paths[key])}")
    if not silent:
        print("")

    # Needed Dirs summary
    if dotenv_loaded_paths and not silent:
        print("-Needed Paths---------------------------------------------------")
    if dotenv_loaded_paths and all_exist_path == False:
        if not silent:
            print("Not all paths were found. Check documentation if you need them")
        retval_paths_exist = False
    if not silent:
        if dotenv_loaded_paths and all_exist_path:
            print("All Needed PATHS exist.")
    if dotenv_needed_models:
        if not silent:
            print("-Needed Models--------------------------------------------------")
        # some model directories were missing
        if none_exist_dir == False and all_exist_dir == False:
            if not silent:
                print("Some manually downloaded models were found. Some might need to be downloaded!")
        # some hf cache models were missing
        if all_exist_hf == False and none_exist_hf == False:
            if not silent:
                print("Some HF_Download models were found. Some might need to be downloaded!")
        if none_exist_dir and none_exist_hf:
            if not silent:
                print("No models were found! Models will be downloaded at next app start")

        if all_exist_hf == True or all_exist_dir == True:
            if not silent:
                print("RESULT: It seems all models were found. Nothing will be downloaded!")
        if all_exist_hf == False and all_exist_dir == False:
            retval_models_exist = False

    retval_final = retval_models_exist == True and retval_paths_exist == True
    return retval_final


def lcx_checkmodels(
    dotenv_needed_models: Set[str],
    dotenv_loaded_paths: Dict[str, str],
    dotenv_loaded_models: Dict[str, str],
    dotenv_loaded_models_values: Dict[str, str],
    in_files_to_check_in_paths: List[Dict[str, str]] = None
) -> None:
    """Check models and exit the program."""
    import sys
    check_do_all_files_exist(
        dotenv_needed_models,
        dotenv_loaded_paths,
        dotenv_loaded_models,
        dotenv_loaded_models_values,
        in_files_to_check_in_paths=in_files_to_check_in_paths
    )
    sys.exit()
