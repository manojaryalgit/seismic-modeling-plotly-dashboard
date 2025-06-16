#!/usr/bin/env python3
"""
Verification script for Nepal Earthquake Analysis Dashboard.
This script verifies essential components of the application.
"""

import os
import importlib

def check_module(module_name):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def check_file(file_path):
    """Check if a file exists"""
    return os.path.isfile(file_path)

def main():
    """Run quick verification checks"""
    print("Running Nepal Earthquake Analysis Dashboard verification...")
    
    # Check core modules
    core_modules = [
        'dash', 'pandas', 'plotly', 'dash_bootstrap_components', 
        'sklearn', 'xgboost', 'catboost', 'gunicorn'
    ]
    
    missing_modules = []
    for module in core_modules:
        if not check_module(module):
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing core modules: {', '.join(missing_modules)}")
        print("Please install them using: pip install -r requirements.txt")
    else:
        print("✅ All core modules are installed")
    
    # Check essential files
    essential_files = [
        'app.py',
        'requirements.txt',
        'pages/__init__.py',
        'pages/overview.py'
    ]
    
    missing_files = []
    for file in essential_files:
        if not check_file(os.path.join(os.path.dirname(__file__), file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing essential files: {', '.join(missing_files)}")
    else:
        print("✅ All essential files are present")
    
    # Summary
    if not (missing_modules or missing_files):
        print("\n✅ Verification passed! Run the app with: python app.py")
    else:
        print("\n❌ Some checks failed. Please address the issues above.")

if __name__ == "__main__":
    main()
