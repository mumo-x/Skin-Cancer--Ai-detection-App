"""
Setup script to install required packages in the correct order.
"""

import subprocess
import sys

def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("Starting installation of required packages...")
    
    # First uninstall potentially conflicting packages
    print("Uninstalling numpy and pandas to avoid version conflicts...")
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "numpy", "pandas"])
    
    # Install packages in the correct order
    packages = [
        "numpy==1.23.5",
        "pandas==1.5.3",
        "pillow==9.3.0",
        "matplotlib==3.6.3",
        "scikit-learn==1.2.2",
        "tensorflow==2.10.0",
        "flask==2.0.1",
        "werkzeug==2.0.1"
    ]
    
    for package in packages:
        install_package(package)
    
    print("\nInstallation completed. You can now run check_env.py to verify the installation.")

if __name__ == "__main__":
    main()