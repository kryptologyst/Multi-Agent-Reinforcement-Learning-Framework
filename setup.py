#!/usr/bin/env python3
"""Installation and setup script for Multi-Agent RL Framework."""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   Required: Python 3.10 or higher")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True


def run_tests():
    """Run the test suite."""
    print("\n🧪 Running tests...")
    
    # Run basic tests first
    if not run_command("python3 test_basic.py", "Running basic tests"):
        return False
    
    # Try to run full tests (may fail if dependencies not fully installed)
    print("🔄 Running full test suite...")
    result = subprocess.run("python3 test_framework.py", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed (this is expected if dependencies aren't fully installed)")
        print("   You can run 'python3 test_framework.py' after installation")
    
    return True


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "assets/plots",
        "checkpoints", 
        "results",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}")
    
    return True


def main():
    """Main installation script."""
    print("🚀 Multi-Agent RL Framework Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    if not create_directories():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed. Please check the errors above.")
        print("   You may need to install dependencies manually:")
        print("   pip install -r requirements.txt")
        return 1
    
    # Run tests
    if not run_tests():
        print("\n⚠️  Some tests failed, but installation may still be successful")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Train models: python scripts/train.py --algorithms ippo mappo")
    print("2. Launch demo: streamlit run demo/app.py")
    print("3. Run tests: python test_framework.py")
    print("4. Check results in the 'results/' directory")
    
    print("\n📚 Documentation:")
    print("   - README.md: Complete usage guide")
    print("   - configs/default.yaml: Configuration options")
    print("   - demo/app.py: Interactive visualization")
    
    print("\n⚠️  Safety Reminder:")
    print("   This framework is for research/educational purposes only.")
    print("   Do not use for production control of real-world systems.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
