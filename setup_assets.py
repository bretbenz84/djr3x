#!/usr/bin/env python3
"""
DJ-R3X Asset Setup Script
Downloads required model files that are too large for git.
Run this once after cloning the repo.
"""
import os
import sys
import urllib.request
import bz2
import shutil
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "assets" / "models"

REQUIRED_MODELS = [
    {
        "name": "Shape Predictor (face landmarks)",
        "filename": "shape_predictor_68_face_landmarks.dat",
        "url": "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2",
        "compressed": True,
        "min_size_mb": 90,
    },
    {
        "name": "MMOD Face Detector",
        "filename": "mmod_human_face_detector.dat",
        "url": "https://github.com/davisking/dlib-models/raw/master/mmod_human_face_detector.dat.bz2",
        "compressed": True,
        "min_size_mb": 0.5,
    },
]

def download_file(url, dest_path):
    print(f"  Downloading {url.split('/')[-1]}...")
    def progress(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 // total_size
            print(f"\r  Progress: {min(pct, 100)}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest_path, reporthook=progress)
    print()

def setup_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for model in REQUIRED_MODELS:
        dest = MODELS_DIR / model["filename"]
        size_mb = dest.stat().st_size / 1024 / 1024 if dest.exists() else 0

        if dest.exists() and size_mb >= model["min_size_mb"]:
            print(f"  OK: {model['filename']} ({size_mb:.1f} MB)")
            continue

        print(f"\nDownloading: {model['name']}")
        compressed_path = MODELS_DIR / (model["filename"] + ".bz2")

        try:
            download_file(model["url"], compressed_path)
            if model["compressed"]:
                print(f"  Decompressing...")
                with bz2.open(compressed_path, "rb") as f_in:
                    with open(dest, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                compressed_path.unlink()
            print(f"  Done: {model['filename']}")
        except Exception as e:
            print(f"  FAILED: {e}")
            all_ok = False

    return all_ok

def fix_face_recognition_models():
    """Fix pkg_resources issue on Python 3.11+ / macOS"""
    import site
    path = Path(site.getsitepackages()[0]) / "face_recognition_models" / "__init__.py"
    if not path.exists():
        return
    content = path.read_text()
    if "pkg_resources" not in content:
        return
    print("\nPatching face_recognition_models for Python 3.11+...")
    new_content = '''# -*- coding: utf-8 -*-
__author__ = """Adam Geitgey"""
__email__ = 'ageitgey@gmail.com'
__version__ = '0.1.0'
import os
_models_dir = os.path.join(os.path.dirname(__file__), "models")
def pose_predictor_model_location():
    return os.path.join(_models_dir, "shape_predictor_68_face_landmarks.dat")
def pose_predictor_five_point_model_location():
    return os.path.join(_models_dir, "shape_predictor_5_face_landmarks.dat")
def face_recognition_model_location():
    return os.path.join(_models_dir, "dlib_face_recognition_resnet_model_v1.dat")
def cnn_face_detector_model_location():
    return os.path.join(_models_dir, "mmod_human_face_detector.dat")
'''
    path.write_text(new_content)
    print("  Patched successfully")

if __name__ == "__main__":
    print("DJ-R3X Asset Setup")
    print("=" * 40)
    fix_face_recognition_models()
    print("\nChecking model files...")
    ok = setup_models()
    print("\n" + ("=" * 40))
    if ok:
        print("Setup complete! All assets ready.")
    else:
        print("Setup completed with errors. Check output above.")
        sys.exit(1)