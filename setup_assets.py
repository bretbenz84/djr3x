#!/usr/bin/env python3
"""
DJ-R3X Asset Setup Script
Downloads required model files that are too large for git.
Run this once after cloning the repo.
"""
import os
import importlib.util
import platform
import sys
import urllib.request
import bz2
import shutil
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "assets" / "models"
XTTS_DIR = MODELS_DIR / "djrex_xtts"
XTTS_DATASET_URL = "https://huggingface.co/buckets/bretbenz/djr3x/resolve/dataset.zip?download=true"
XTTS_DATASET_PATH = XTTS_DIR / "dataset.zip"
XTTS_MIN_SIZE_MB = 1.0
DOTENV_PATH = Path(__file__).parent / ".env"

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
    {
        "name": "dlib Face Recognition ResNet Model",
        "filename": "dlib_face_recognition_resnet_model_v1.dat",
        "url": "https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2",
        "compressed": True,
        "min_size_mb": 20,
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


def _is_macos_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _selected_tts_provider() -> str:
    if not DOTENV_PATH.exists():
        return ""
    try:
        for raw_line in DOTENV_PATH.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "TTS_PROVIDER":
                return value.strip().strip("\"'").lower()
    except OSError:
        return ""
    return ""


def setup_xtts_dataset() -> bool:
    """Download the optional XTTS dataset zip for Apple Silicon setups."""
    if not _is_macos_apple_silicon():
        print("\nSkipping XTTS dataset download (macOS Apple Silicon only).")
        return True

    XTTS_DIR.mkdir(parents=True, exist_ok=True)
    size_mb = XTTS_DATASET_PATH.stat().st_size / 1024 / 1024 if XTTS_DATASET_PATH.exists() else 0
    if XTTS_DATASET_PATH.exists() and size_mb >= XTTS_MIN_SIZE_MB:
        print(f"\n  OK: {XTTS_DATASET_PATH.name} ({size_mb:.1f} MB)")
        return True

    print("\nDownloading: XTTS dataset.zip (Apple Silicon only)")
    tmp_path = XTTS_DATASET_PATH.with_suffix(".zip.part")
    try:
        download_file(XTTS_DATASET_URL, tmp_path)
        tmp_path.replace(XTTS_DATASET_PATH)
        size_mb = XTTS_DATASET_PATH.stat().st_size / 1024 / 1024
        print(f"  Done: {XTTS_DATASET_PATH.name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        return False


def check_xtts_runtime() -> bool:
    """Verify Apple Silicon XTTS Python dependencies when XTTS is selected."""
    if not _is_macos_apple_silicon():
        return True

    provider = _selected_tts_provider()
    if provider != "xtts":
        print("\nSkipping XTTS Python package check (TTS_PROVIDER is not xtts).")
        return True

    required_modules = ("torch", "torchaudio", "TTS")
    missing = [
        module_name
        for module_name in required_modules
        if importlib.util.find_spec(module_name) is None
    ]

    print("\nChecking XTTS Python dependencies (Apple Silicon)")
    print("-" * 40)
    if not missing:
        print("  OK: torch, torchaudio, and TTS are installed")
        return True

    print(f"  MISSING: {', '.join(missing)}")
    print("  Install with: pip install -r requirements-macos-apple-silicon.txt")
    return False


def report_xtts_status():
    """Report whether the optional Apple Silicon XTTS assets are present."""
    vocab_candidates = [
        XTTS_DIR / "vocab.json",
        XTTS_DIR / "vocab.json_",
    ]
    vocab_path = next((p for p in vocab_candidates if p.exists()), vocab_candidates[0])

    print("\nOptional XTTS voice (Apple Silicon only)")
    print("-" * 40)
    checks = [
        ("XTTS config", XTTS_DIR / "config.json"),
        ("XTTS checkpoint", XTTS_DIR / "model.pth"),
        ("XTTS vocab", vocab_path),
        ("XTTS dataset zip", XTTS_DATASET_PATH),
        ("XTTS speaker reference", Path(__file__).parent / "reference.wav"),
    ]
    for label, path in checks:
        exists = path.exists()
        print(f"  {'OK' if exists else 'MISSING'}: {label}: {path}")
    print("  Note: XTTS runs only on macOS Apple Silicon and is selected with TTS_PROVIDER=xtts")

def fix_face_recognition_models():
    """Fix pkg_resources issue if face_recognition_models is installed."""
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
    ok = setup_xtts_dataset() and ok
    ok = check_xtts_runtime() and ok
    report_xtts_status()
    print("\n" + ("=" * 40))
    if ok:
        print("Setup complete! All assets ready.")
    else:
        print("Setup completed with errors. Check output above.")
        sys.exit(1)
