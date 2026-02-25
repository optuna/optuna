#!/usr/bin/env python3
"""Test script to verify that unused type ignore comments have been removed."""

import subprocess
import sys

def run_mypy_check():
    """Run mypy on specific files that had unused-ignore errors."""
    files_to_check = [
        "optuna/visualization/matplotlib/_timeline.py",
        "optuna/visualization/matplotlib/_rank.py",
        "optuna/_gp/gp.py",
        "optuna/_gp/acqf.py",
        "optuna/_gp/optim_mixed.py",
        "optuna/storages/journal/_redis.py",
        "optuna/samplers/_grid.py",
        "optuna/storages/_rdb/models.py"
    ]
    
    for file_path in files_to_check:
        print(f"Checking {file_path} for unused-ignore errors...")
        result = subprocess.run(
            ["mypy", file_path, "--strict"],
            capture_output=True,
            text=True
        )
        
        # Check if there are any unused-ignore errors specifically
        unused_ignore_errors = [line for line in result.stdout.split('\n') 
                               if 'unused-ignore' in line.lower()]
        
        if unused_ignore_errors:
            print(f"❌ Found unused-ignore errors in {file_path}:")
            for error in unused_ignore_errors:
                print(f"  {error}")
            return False
        else:
            print(f"✅ No unused-ignore errors found in {file_path}")
    
    print("All specified files pass the unused-ignore check!")
    return True

if __name__ == "__main__":
    if run_mypy_check():
        print("✅ Fix successful: All unused type ignore comments have been removed!")
        sys.exit(0)
    else:
        print("❌ Fix incomplete: Some unused type ignore comments remain.")
        sys.exit(1)