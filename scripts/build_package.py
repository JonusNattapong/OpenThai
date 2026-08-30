"""
Build Wheel and Source Distribution for PyPI Release.
Validates package metadata with twine.
"""

import os
import shutil
import subprocess
import sys


def clean():
    print("[1/3] Cleaning previous build artifacts...")
    for folder in ["build", "dist", "openthai_ner.egg-info"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"  Removed {folder}/")


def build():
    print("\n[2/3] Building distribution packages (.whl & .tar.gz)...")
    # Ensure build and twine are installed
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "build", "twine"])
    subprocess.check_call([sys.executable, "-m", "build"])


def validate():
    print("\n[3/3] Validating package metadata with twine...")
    subprocess.check_call([sys.executable, "-m", "twine", "check", "dist/*"])
    print("\n" + "=" * 65)
    print("[SUCCESS] PACKAGE BUILT AND VALIDATED SUCCESSFULLY!")
    print("=" * 65)
    print("Generated files in dist/:")
    for f in os.listdir("dist"):
        print(f"  - dist/{f}")
    print("\nTo publish to PyPI, run:")
    print("  twine upload dist/*")
    print("=" * 65)


if __name__ == "__main__":
    clean()
    build()
    validate()
