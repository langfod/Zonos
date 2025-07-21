"""
Configuration utilities for managing model paths and config files.
"""
import os
import re
from pathlib import Path
from typing import Dict, Set, Any, Tuple


def model_to_varname(model_path: str, prefix: str) -> str:
    """Converts a model path to a dotenv-compatible variable name"""
    model_name = model_path.split("/")[-1]
    varname = re.sub(r"[^a-zA-Z0-9]", "_", model_name.upper())
    return f"{prefix}{varname}"


def varname_to_model(varname: str, prefix: str) -> str:
    """Converts a variable name back to original model path format"""
    if varname.startswith("PATH_MODEL_"):
        model_part = varname[len(prefix):].lower().replace("_", "-")
        return f"Zyphra/{model_part}"
    return ""


def read_existing_config(file_path: str) -> Dict[str, str]:
    """Reads existing config file and returns key-value pairs"""
    existing = {}
    path = Path(file_path)
    if path.exists():
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        existing[parts[0].strip()] = parts[1].strip()
    else:
        print(f"ERROR config file not found: {file_path}")
    return existing


def update_model_paths_file(
    models: Set[str],
    paths: Dict[str, str],
    params: Dict[str, Any],
    file_path: str,
    prefix_model: str = "PATH_MODEL_",
    prefix_path: str = "PATH_NEEDED_",
    log_prefix: str = "CROSSOS_LOG"
) -> None:
    """Updates config file, adding only new variables"""
    existing = read_existing_config(file_path)
    new_lines = []

    # Process models
    for model in models:
        varname = model_to_varname(model, prefix_model)
        if varname not in existing:
            print(f"{log_prefix}: Adding Model requirement to config: {model}")
            new_lines.append(f"{varname} = ./models/{model.split('/')[-1]}")

    # Process paths - now handles any path keys
    for key, value in paths.items():
        varname = model_to_varname(key, prefix_path)
        if varname not in existing:
            print(f"{log_prefix}: Adding path requirement to config: {key}")
            new_lines.append(f"{varname} = {value}")

    # Process params
    for key, value in params.items():
        if key not in existing:
            print(f"{log_prefix}: Adding Parameter requirement to config: {key}")
            new_lines.append(f"{key} = {value}")

    # Append new lines if any
    if new_lines:
        with open(file_path, 'a') as f:
            f.write("\n" + "\n".join(new_lines) + "\n")


def parse_model_paths_file(
    file_path: str,
    dotenv_needed_models: Set[str],
    dotenv_needed_paths: Dict[str, str],
    prefix_model: str = "PATH_MODEL_",
    prefix_path: str = "PATH_NEEDED_"
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any], Dict[str, str]]:
    """Reads config file and returns loaded variables"""
    loaded_models = {}
    loaded_paths = {}
    loaded_params = {}
    loaded_models_values = {}
    existing = read_existing_config(file_path)

    for key, value in existing.items():
        # Handle model paths
        if key.startswith(prefix_model):
            for mod in dotenv_needed_models:
                # we find out if the current key value belongs to one of our models
                if key == model_to_varname(mod, prefix_model):
                    # if a path has been defined and it exists we use the local path
                    if value and os.path.isdir(value):
                        loaded_models[mod] = value
                    else:
                        # else we use the model id so its downloaded from github later
                        loaded_models[mod] = mod
                    # still we collect the values to show to the user so he knows what to fix in config file
                    loaded_models_values[mod] = value
        # Handle ALL paths (not just HF_HOME)
        elif key.startswith(prefix_path):
            for mod in dotenv_needed_paths:
                if key == model_to_varname(mod, prefix_path):
                    loaded_paths[mod] = value
        # Handle params with type conversion
        else:
            if value.lower() in {"true", "false"}:
                loaded_params[key] = value.lower() == "true"
            elif value.isdigit():
                loaded_params[key] = int(value)
            else:
                try:
                    loaded_params[key] = float(value)
                except ValueError:
                    loaded_params[key] = value

    return loaded_models, loaded_paths, loaded_params, loaded_models_values


def is_online_model(model: str, dotenv_needed_models: Set[str], debug_mode: bool = False) -> bool:
    """Checks if a model is in the online models set."""
    is_onlinemodel = model in dotenv_needed_models
    if debug_mode:
        print(f"Model '{model}' is online: {is_onlinemodel}")
    return is_onlinemodel
