#!/usr/bin/env python3
"""
Script to convert all manifest files and schema to use snake_case consistently.
"""

import os
import sys
import yaml
import re
from pathlib import Path
from typing import Any, Dict, List, Union

def to_snake_case(name: str) -> str:
    """Convert camelCase or PascalCase to snake_case."""
    # Handle special cases first
    special_cases = {
        'apiVersion': 'api_version',
        'modelType': 'model_type',
        'modelTypes': 'model_types',
        'engineType': 'engine_type',
        'denoiseType': 'denoise_type',
        'config_path': 'config_path',  # already snake_case
        'model_path': 'model_path',    # already snake_case
        'file_pattern': 'file_pattern', # already snake_case
        'key_map': 'key_map',          # already snake_case
        'extra_kwargs': 'extra_kwargs', # already snake_case
        'save_path': 'save_path',      # already snake_case
        'converter_kwargs': 'converter_kwargs', # already snake_case
        'model_key': 'model_key',      # already snake_case
        'extra_model_paths': 'extra_model_paths', # already snake_case
        'converted_model_path': 'converted_model_path', # already snake_case
        'num_inference_steps': 'num_inference_steps', # already snake_case
        'guidance_scale': 'guidance_scale', # already snake_case
        'return_latents': 'return_latents', # already snake_case
        'text_encoder_kwargs': 'text_encoder_kwargs', # already snake_case
        'attention_kwargs': 'attention_kwargs', # already snake_case
        'boundary_ratio': 'boundary_ratio', # already snake_case
        'expand_timesteps': 'expand_timesteps', # already snake_case
        'use_mask_in_input': 'use_mask_in_input', # already snake_case
        'pad_with_zero': 'pad_with_zero', # already snake_case
        'max_sequence_length': 'max_sequence_length', # already snake_case
        'use_token_type_ids': 'use_token_type_ids', # already snake_case
        'use_position_ids': 'use_position_ids', # already snake_case
        'clean_text': 'clean_text', # already snake_case
        'guidance_rescale': 'guidance_rescale', # already snake_case
        'render_on_step': 'render_on_step', # already snake_case
    }
    
    if name in special_cases:
        return special_cases[name]
    
    # If already snake_case, return as-is
    if '_' in name and name.islower():
        return name
    
    # Convert camelCase/PascalCase to snake_case
    # Insert underscore before uppercase letters that follow lowercase letters
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Insert underscore before uppercase letters that follow lowercase letters or digits
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def convert_dict_keys(obj: Any) -> Any:
    """Recursively convert all dictionary keys to snake_case."""
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            new_key = to_snake_case(key)
            new_dict[new_key] = convert_dict_keys(value)
        return new_dict
    elif isinstance(obj, list):
        return [convert_dict_keys(item) for item in obj]
    else:
        return obj

def process_yaml_file(file_path: Path):
    """Process a single YAML file to convert keys to snake_case."""
    print(f"Processing {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        
        if data is None:
            print(f"  Skipping empty file")
            return
        
        # Convert all keys to snake_case
        converted_data = convert_dict_keys(data)
        
        # Write back to file
        with open(file_path, 'w') as f:
            yaml.dump(converted_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        print(f"  Converted successfully")
        
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")

def process_python_file(file_path: Path):
    """Process a Python file to convert schema keys to snake_case."""
    print(f"Processing {file_path}...")
    
    try:
        content = file_path.read_text()
        
        # Dictionary of replacements for schema keys
        replacements = {
            '"apiVersion"': '"api_version"',
            '"modelType"': '"model_type"',
            '"modelTypes"': '"model_types"',
            '"engineType"': '"engine_type"',
            '"denoiseType"': '"denoise_type"',
            '"additionalProperties"': '"additional_properties"',
            '"minLength"': '"min_length"',
            '"maxLength"': '"max_length"',
            '"minItems"': '"min_items"',
            '"maxItems"': '"max_items"',
            "'apiVersion'": "'api_version'",
            "'modelType'": "'model_type'",
            "'modelTypes'": "'model_types'",
            "'engineType'": "'engine_type'",
            "'denoiseType'": "'denoise_type'",
            "'additionalProperties'": "'additional_properties'",
            "'minLength'": "'min_length'",
            "'maxLength'": "'max_length'",
            "'minItems'": "'min_items'",
            "'maxItems'": "'max_items'",
        }
        
        # Apply replacements
        modified = False
        for old, new in replacements.items():
            if old in content:
                content = content.replace(old, new)
                modified = True
        
        if modified:
            file_path.write_text(content)
            print(f"  Updated schema keys")
        else:
            print(f"  No changes needed")
            
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")

def main():
    base_dir = Path('/Users/tosinkuye/apex')
    
    print("=== Converting to snake_case ===")
    
    # Process all YAML files in manifest directory
    print("\n1. Processing manifest YAML files...")
    manifest_dir = base_dir / 'manifest'
    
    if manifest_dir.exists():
        for yaml_file in manifest_dir.rglob('*.yml'):
            if yaml_file.is_file():
                process_yaml_file(yaml_file)
        
        for yaml_file in manifest_dir.rglob('*.yaml'):
            if yaml_file.is_file():
                process_yaml_file(yaml_file)
    
    # Process schema files
    print("\n2. Processing schema files...")
    src_manifest_dir = base_dir / 'src' / 'manifest'
    
    if src_manifest_dir.exists():
        for py_file in src_manifest_dir.rglob('*.py'):
            if py_file.is_file() and 'schema' in py_file.name:
                process_python_file(py_file)
    
    # Process loader files that might reference schema keys
    print("\n3. Processing loader files...")
    loader_files = [
        base_dir / 'src' / 'manifest' / 'loader.py',
        base_dir / 'src' / 'manifest' / 'shared_loader.py',
    ]
    
    for loader_file in loader_files:
        if loader_file.exists():
            process_python_file(loader_file)
    
    print("\n=== Conversion Complete ===")

if __name__ == '__main__':
    main()
