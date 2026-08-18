# core/config.py
import os
import sys
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "audio"
OUTPUT_DIR = BASE_DIR / "output"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# NVIDIA CUDA DLL 路徑配置
VENV_DIR = BASE_DIR / "venv"
NVIDIA_SITE_PACKAGES = VENV_DIR / "Lib" / "site-packages" / "nvidia"

CUDA_DLL_DIRS = [
    NVIDIA_SITE_PACKAGES / "cublas" / "bin",
    NVIDIA_SITE_PACKAGES / "cudnn" / "bin",
    NVIDIA_SITE_PACKAGES / "cuda_nvrtc" / "bin",
]

def setup_cuda_env():
    dll_paths = [str(p) for p in CUDA_DLL_DIRS if p.exists()]
    if dll_paths:
        os.environ["PATH"] = ";".join(dll_paths) + ";" + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            for p in dll_paths:
                try:
                    os.add_dll_directory(p)
                except OSError:
                    pass

setup_cuda_env()