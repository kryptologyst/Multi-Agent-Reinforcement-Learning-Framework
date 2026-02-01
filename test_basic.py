#!/usr/bin/env python3
"""Minimal test script that doesn't require external dependencies."""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_structure():
    """Test basic project structure."""
    print("Testing project structure...")
    
    required_dirs = [
        'src',
        'src/algorithms',
        'src/envs', 
        'src/eval',
        'src/utils',
        'configs',
        'scripts',
        'demo',
        'tests',
        'assets'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} missing")
            return False
    
    return True


def test_config_files():
    """Test configuration files."""
    print("\nTesting configuration files...")
    
    config_files = [
        'requirements.txt',
        'pyproject.toml',
        'README.md',
        'configs/default.yaml',
        '.gitignore'
    ]
    
    for file_path in config_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True


def test_code_quality():
    """Test basic code quality (syntax)."""
    print("\nTesting code quality...")
    
    python_files = [
        'src/envs/grid_world.py',
        'src/algorithms/ippo.py',
        'src/algorithms/mappo.py',
        'src/eval/evaluator.py',
        'scripts/train.py',
        'demo/app.py',
        'tests/test_components.py'
    ]
    
    for file_path in python_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    compile(f.read(), file_path, 'exec')
                print(f"✅ {file_path} syntax OK")
            except SyntaxError as e:
                print(f"❌ {file_path} syntax error: {e}")
                return False
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True


def test_import_structure():
    """Test import structure without external dependencies."""
    print("\nTesting import structure...")
    
    # Test that __init__.py files exist
    init_files = [
        'src/__init__.py',
        'src/algorithms/__init__.py',
        'src/envs/__init__.py',
        'src/eval/__init__.py',
        'src/utils/__init__.py'
    ]
    
    for init_file in init_files:
        if os.path.exists(init_file):
            print(f"✅ {init_file} exists")
        else:
            print(f"❌ {init_file} missing")
            return False
    
    return True


def test_documentation():
    """Test documentation completeness."""
    print("\nTesting documentation...")
    
    # Check README has key sections
    if os.path.exists('README.md'):
        with open('README.md', 'r') as f:
            readme_content = f.read()
        
        key_sections = [
            'Installation',
            'Quick Start',
            'Safety Disclaimer',
            'Features',
            'Usage'
        ]
        
        for section in key_sections:
            if section in readme_content:
                print(f"✅ README contains {section}")
            else:
                print(f"❌ README missing {section}")
                return False
    else:
        print("❌ README.md missing")
        return False
    
    return True


def main():
    """Run all basic tests."""
    print("🧪 Multi-Agent RL Framework - Basic Test Suite")
    print("=" * 60)
    print("Note: This test doesn't require external dependencies")
    print("=" * 60)
    
    tests = [
        test_basic_structure,
        test_config_files,
        test_code_quality,
        test_import_structure,
        test_documentation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Basic Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! Project structure is correct.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full tests: python test_framework.py")
        print("3. Start training: python scripts/train.py --algorithms ippo")
        print("4. Launch demo: streamlit run demo/app.py")
        return 0
    else:
        print("❌ Some basic tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
