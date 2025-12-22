#!/bin/bash

# ============================================================================
# SoundFlow Premium GPU Setup Script
# ============================================================================
# This script sets up the environment for premium AI music generation
# Supports: Ubuntu/Debian, Google Colab, Kaggle
# ============================================================================

set -e  # Exit on error

echo "🎵 SoundFlow Premium GPU Setup"
echo "================================"
echo ""

# ============================================================================
# STEP 1: Check GPU
# ============================================================================

echo "🔍 Checking GPU availability..."

if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  Warning: nvidia-smi not found. GPU may not be available."
    echo "   Make sure CUDA drivers are installed."
fi

echo ""

# ============================================================================
# STEP 2: Install System Dependencies
# ============================================================================

echo "📦 Installing system dependencies..."

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    OS="unknown"
fi

case $OS in
    ubuntu|debian)
        echo "   Detected Ubuntu/Debian"
        sudo apt-get -qq update
        sudo apt-get -qq install -y ffmpeg
        ;;
    *)
        echo "   Unknown OS: $OS"
        echo "   Please install FFmpeg manually"
        ;;
esac

# Verify FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo "✅ FFmpeg installed: $(ffmpeg -version | head -n1)"
else
    echo "❌ FFmpeg installation failed!"
    exit 1
fi

echo ""

# ============================================================================
# STEP 3: Install Python Dependencies
# ============================================================================

echo "🐍 Installing Python dependencies..."

# Check if we're in generator directory
if [ ! -f "requirements-gpu.txt" ]; then
    echo "❌ Error: requirements-gpu.txt not found!"
    echo "   Make sure you're running this from the generator/ directory"
    exit 1
fi

# Install dependencies
pip install -q -r requirements-gpu.txt

echo "✅ Python dependencies installed"
echo ""

# ============================================================================
# STEP 4: Verify Installation
# ============================================================================

echo "🧪 Verifying installation..."

python3 << 'EOF'
import sys

def check_import(name, package=None):
    package = package or name
    try:
        __import__(package)
        print(f"  ✅ {name}")
        return True
    except ImportError:
        print(f"  ❌ {name} - FAILED")
        return False

print("\nChecking core dependencies:")
all_ok = True
all_ok &= check_import("torch")
all_ok &= check_import("audiocraft")
all_ok &= check_import("scipy")
all_ok &= check_import("numpy")
all_ok &= check_import("ffmpeg-python", "ffmpeg")

print("\nChecking GPU:")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✅ GPU: {gpu_name} ({gpu_mem:.1f}GB VRAM)")

        if gpu_mem < 15:
            print(f"  ⚠️  Warning: Less than 15GB VRAM detected.")
            print(f"     Consider using musicgen-stereo-medium instead of large")
    else:
        print("  ⚠️  CUDA not available. CPU mode only.")
        all_ok = False
except Exception as e:
    print(f"  ❌ GPU check failed: {e}")
    all_ok = False

if not all_ok:
    print("\n❌ Some dependencies failed. Please fix errors above.")
    sys.exit(1)
else:
    print("\n✅ All checks passed!")
EOF

echo ""

# ============================================================================
# STEP 5: Create Output Directories
# ============================================================================

echo "📁 Creating output directories..."

mkdir -p .soundflow_out/premium
mkdir -p .soundflow_tmp/premium

echo "✅ Directories created"
echo ""

# ============================================================================
# DONE
# ============================================================================

echo "================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo ""
echo "  1. Test generation:"
echo "     python -m premium.musicgen_daily --date $(date +%Y-%m-%d) --duration 30"
echo ""
echo "  2. Generate all premium tracks:"
echo "     python -m premium.musicgen_daily --date $(date +%Y-%m-%d)"
echo ""
echo "  3. Use medium model (less VRAM):"
echo "     python -m premium.musicgen_daily --date $(date +%Y-%m-%d) \\"
echo "       --model facebook/musicgen-stereo-medium"
echo ""
echo "📚 See PREMIUM_README.md for full documentation"
echo ""
