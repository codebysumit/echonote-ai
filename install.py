"""
EchoNote AI — Smart Dependency Installer
─────────────────────────────────────────────────────────────────
Automatically detects your hardware and installs the correct
llama-cpp-python wheel + faster-whisper backend:

  NVIDIA CUDA 12.x  →  llama-cpp-python CUDA wheel  (GPU enabled)
  NVIDIA CUDA 11.x  →  llama-cpp-python CUDA 11 wheel
  Apple MPS         →  llama-cpp-python Metal wheel  (M1/M2/M3)
  No GPU / Unknown  →  llama-cpp-python CPU-only wheel

Run:
    python install.py
"""

import subprocess
import sys
import platform

# ── helpers ──────────────────────────────────────────────────────────────────

def run(cmd: str, desc: str = ""):
    print(f"\n{'─'*60}")
    if desc:
        print(f"  → {desc}")
    print(f"  $ {cmd}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n⚠️  Command exited with code {result.returncode} — continuing anyway...\n")

def pip(*packages, extra_index: str = ""):
    cmd = f"{sys.executable} -m pip install {' '.join(packages)}"
    if extra_index:
        cmd += f" --extra-index-url {extra_index}"
    run(cmd)

# ── Hardware Detection ────────────────────────────────────────────────────────

def detect_cuda_version() -> tuple:
    """
    Returns (major, minor) CUDA version, e.g. (12, 1) for CUDA 12.1
    Returns (0, 0) if no CUDA GPU found.
    """
    try:
        import torch
        if torch.cuda.is_available():
            version_str = torch.version.cuda  # e.g. "12.1"
            parts = version_str.split(".")
            return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except ImportError:
        pass

    # Fallback: try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            # nvidia-smi found a GPU but we can't tell CUDA version without torch
            # Try nvcc
            nvcc = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=5
            )
            if nvcc.returncode == 0:
                for line in nvcc.stdout.splitlines():
                    if "release" in line.lower():
                        # e.g. "Cuda compilation tools, release 12.1, V12.1.105"
                        parts = line.split("release")[-1].strip().split(",")[0].strip()
                        major, minor = parts.split(".")[:2]
                        return int(major), int(minor)
            return 12, 1  # assume CUDA 12.1 as safe default
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return 0, 0  # no CUDA

def is_apple_silicon() -> bool:
    """Returns True on Apple M-chip (MPS) systems."""
    return platform.system() == "Darwin" and platform.processor() == "arm"

# ── Wheel Selection ───────────────────────────────────────────────────────────

CUDA_WHEEL_MAP = {
    (12, 6): "cu126",
    (12, 5): "cu125",
    (12, 4): "cu124",
    (12, 3): "cu123",
    (12, 2): "cu122",
    (12, 1): "cu121",
    (12, 0): "cu120",
    (11, 8): "cu118",
    (11, 7): "cu117",
}
BASE_WHEEL_URL = "https://abetlen.github.io/llama-cpp-python/whl"

def get_llama_cpp_wheel(cuda_major: int, cuda_minor: int) -> str:
    """Returns the correct --extra-index-url for llama-cpp-python."""
    if cuda_major == 0:
        if is_apple_silicon():
            return f"{BASE_WHEEL_URL}/metal"  # Apple Metal GPU
        return f"{BASE_WHEEL_URL}/cpu"        # CPU-only fallback

    # Find closest supported CUDA wheel
    for (maj, min_), tag in CUDA_WHEEL_MAP.items():
        if cuda_major == maj and cuda_minor >= min_:
            return f"{BASE_WHEEL_URL}/{tag}"

    # Fallback to closest major
    if cuda_major >= 12:
        return f"{BASE_WHEEL_URL}/cu126"
    return f"{BASE_WHEEL_URL}/cu118"

# ── Main Install Flow ─────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*60)
    print("  🎙️  EchoNote AI — Smart Installer")
    print("═"*60)

    # Step 1: Core dependencies (always the same)
    print("\n[1/4] Installing core dependencies...")
    pip("fastapi", "uvicorn[standard]", "cryptography", "pydantic",
        "python-multipart", "python-dotenv",
        desc="FastAPI + encryption core")

    # Step 2: faster-whisper (CPU/GPU via CUDA support in PyTorch)
    print("\n[2/4] Installing faster-whisper (auto GPU/CPU)...")
    pip("faster-whisper", "huggingface_hub",
        desc="faster-whisper transcription engine")

    # Step 3: Detect hardware and install correct llama-cpp-python wheel
    print("\n[3/4] Detecting hardware for llama-cpp-python...")
    cuda_major, cuda_minor = detect_cuda_version()
    apple_mps = is_apple_silicon()

    if cuda_major > 0:
        backend_label = f"NVIDIA CUDA {cuda_major}.{cuda_minor}"
    elif apple_mps:
        backend_label = "Apple Silicon (Metal MPS)"
    else:
        backend_label = "CPU-only (no GPU detected)"

    wheel_url = get_llama_cpp_wheel(cuda_major, cuda_minor)
    print(f"\n  ✅ Detected backend : {backend_label}")
    print(f"  ✅ Installing wheel : {wheel_url}")

    pip("llama-cpp-python", extra_index=wheel_url,
        desc=f"llama-cpp-python for {backend_label}")

    # Step 4: Done
    print("\n" + "═"*60)
    print("  ✅  Installation complete!")
    print(f"      Hardware : {backend_label}")
    if cuda_major > 0:
        print(f"      Gemma will use ALL GPU layers (n_gpu_layers=-1)")
    elif apple_mps:
        print(f"      Gemma will use Apple Metal partial GPU offload")
    else:
        print(f"      Gemma will run on CPU (int4 Q4_K_M, ~800 MB RAM)")
    print("\n  Run the app:  python app.py")
    print("  Then open:    http://localhost:8000")
    print("═"*60 + "\n")

if __name__ == "__main__":
    main()
