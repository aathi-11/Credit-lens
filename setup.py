"""
setup.py — One-shot setup: installs dependencies and trains the model.
Run this once before starting the Streamlit app.
"""
import subprocess
import sys
import os

ROOT = os.path.dirname(__file__)


def run(cmd):
    print(f"\n▶  {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.join(ROOT, "src"))
    if result.returncode != 0:
        print(f"❌ Command failed with code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("   Credit Risk Scorer — Setup & Train")
    print("=" * 60)

    # 1. Install requirements
    print("\n📦 Installing dependencies...")
    run([sys.executable, "-m", "pip", "install", "-r", os.path.join(ROOT, "requirements.txt")])

    # 2. Train model
    print("\n🚀 Training model...")
    run([sys.executable, os.path.join(ROOT, "src", "train.py")])

    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("   Run:  streamlit run app.py")
    print("=" * 60)
