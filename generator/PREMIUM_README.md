# 🎵 SoundFlow Premium AI Music Production

**Enterprise-Grade AI Music Generation with Neuro-Symbolic DSP**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![GPU](https://img.shields.io/badge/GPU-CUDA-green.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## 🚀 What's New in Premium

The Premium upgrade transforms SoundFlow from a prototype into **production-ready AI music service** with:

### 🎼 Core Engine Upgrades

| Feature | Free Version | Premium Version |
|---------|--------------|-----------------|
| **AI Model** | Simple MIDI synthesis | **MusicGen Stereo Large** (3.3B params) |
| **Audio Quality** | Mono, basic synthesis | **True Stereo, 32kHz** |
| **Generation Library** | Transformers pipeline | **Native Audiocraft** (full control) |
| **Output Format** | 128kbps MP3 | **320kbps MP3, -14 LUFS** |

### 🧠 Neuro-Symbolic DSP Engine

Premium includes mathematical audio enhancements that AI alone cannot provide:

1. **Binaural Beat Injection**
   - Gamma (40Hz): Peak focus and concentration
   - Beta (20Hz): Active thinking and alertness
   - Alpha (10Hz): Relaxed creativity
   - Theta (6Hz): Deep meditation
   - Delta (3Hz): Deep sleep and recovery

2. **Sidechain Compression**
   - EDM-style "pumping" effect
   - Configurable BPM and intensity
   - Adds energy to House/Techno tracks

3. **Professional Mastering**
   - EBU R128 loudness normalization (-14 LUFS)
   - True peak limiting (-1.5 dBTP)
   - Broadcast-quality output

---

## 📋 Hardware Requirements

### Minimum (Testing)
- GPU: NVIDIA T4 (16GB VRAM) - **Google Colab Free**
- RAM: 12GB
- Disk: 20GB free
- Generation time: ~2-3 min/track

### Recommended (Production)
- GPU: NVIDIA A100 (40GB VRAM) - **Colab Pro, Kaggle**
- RAM: 32GB
- Disk: 50GB+ SSD
- Generation time: ~30-60 sec/track

### Supported Platforms
✅ **Google Colab** (Free/Pro)
✅ **Kaggle Notebooks** (Free GPU)
✅ **RunPod** ($0.30/hour)
✅ **AWS p3/p4 instances**
✅ **Local GPU** (RTX 3090/4090, A6000)

---

## 🛠️ Installation

### Option 1: Google Colab (Recommended for Beginners)

1. Open the pre-configured notebook:
   - [SoundFlow_Premium_Colab.ipynb](notebooks/SoundFlow_Premium_Colab.ipynb)

2. Enable GPU:
   - `Runtime > Change runtime type > Hardware accelerator > GPU`

3. Run all cells (it's that easy!)

### Option 2: Kaggle

1. Create new notebook with GPU enabled
2. Upload `generator/` folder as dataset
3. Install dependencies:
   ```python
   !pip install -q -r /kaggle/input/soundflow/requirements-gpu.txt
   ```

### Option 3: Local GPU

```bash
# Clone repository
git clone https://github.com/ruslanmv/soundflow.git
cd soundflow/generator

# Install dependencies
pip install -r requirements-gpu.txt

# Install FFmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg
# macOS:
brew install ffmpeg
```

---

## 🎹 Quick Start

### Generate Daily Premium Tracks

```bash
# Generate all tracks from premium_prompts.yaml
python -m premium.musicgen_daily --date 2025-01-15 --device cuda

# Use medium model (faster, less VRAM)
python -m premium.musicgen_daily \
  --date 2025-01-15 \
  --model facebook/musicgen-stereo-medium \
  --device cuda

# Override duration to 2 minutes
python -m premium.musicgen_daily \
  --date 2025-01-15 \
  --duration 120 \
  --device cuda
```

Output: `.soundflow_out/premium/premium-2025-01-15-*.mp3`

### Using Makefile (Local Development)

```bash
# Test GPU setup
make gpu-test

# Generate premium tracks
make gpu-premium

# Show Colab setup instructions
make gpu-colab

# Show Kaggle setup instructions
make gpu-kaggle
```

---

## 🎨 Creating Custom Tracks

### 1. Edit Configuration File

Edit `prompts/premium_prompts.yaml`:

```yaml
tracks:
  - id: my_custom_track
    title: "My Custom Focus Track"
    category: "Deep Work"
    prompt: "ambient electronic focus music, minimal beats, warm pads, meditative"
    bpm: 90
    key: "D minor"
    target_total_sec: 300  # 5 minutes

    # DSP Enhancements
    dsp_flags:
      # Binaural beats for focus
      binaural: true
      binaural_freq: 40.0   # Gamma waves
      binaural_base: 200.0
      binaural_mix: 0.15

      # Fades
      fade_in_ms: 2000
      fade_out_ms: 4000
```

### 2. Generate

```bash
python -m premium.musicgen_daily --date 2025-01-15
```

---

## 🔬 DSP Configuration Reference

### Binaural Beat Frequencies

| Wave Type | Frequency | Effect | Use Case |
|-----------|-----------|--------|----------|
| **Gamma** | 30-50 Hz | Peak focus | Coding, studying, deep work |
| **Beta** | 13-30 Hz | Active thinking | Problem solving, analysis |
| **Alpha** | 8-13 Hz | Relaxed creativity | Writing, design, ideation |
| **Theta** | 4-8 Hz | Deep meditation | Meditation, introspection |
| **Delta** | 0.5-4 Hz | Deep sleep | Sleep, recovery, healing |

### DSP Flags Complete Reference

```yaml
dsp_flags:
  # Binaural Beats
  binaural: true                # Enable binaural beats
  binaural_freq: 40.0           # Beat frequency (1-100 Hz)
  binaural_base: 200.0          # Carrier frequency (50-500 Hz)
  binaural_mix: 0.12            # Mix level (0.0-0.5, default 0.12)

  # Sidechain Compression
  sidechain: true               # Enable sidechain
  bpm: 128                      # Tempo for sidechain timing
  ducking_strength: 0.7         # Intensity (0.0-1.0)
  sidechain_attack: 0.3         # Attack ratio (0.1-0.5)

  # Fades
  fade_in_ms: 1500              # Fade-in duration (ms)
  fade_out_ms: 3000             # Fade-out duration (ms)
```

---

## 🌐 API Usage

### Start API Server

```bash
cd generator
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Generate Premium Track via API

```bash
curl -X POST "http://localhost:8000/api/generate/premium" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "uplifting progressive house, 126 BPM, emotional leads",
    "duration": 180,
    "bpm": 126,
    "key": "A major",
    "dsp_flags": {
      "sidechain": true,
      "bpm": 126,
      "ducking_strength": 0.65
    },
    "export_bitrate": "320k",
    "target_lufs": -14.0
  }'
```

### Premium UI

1. Start UI dev server:
   ```bash
   cd generator/ui
   npm install
   npm run dev
   ```

2. Open http://localhost:3000

3. Use Premium Studio tab to:
   - Select category presets (Deep Work, Meditation, Energy, etc.)
   - Configure binaural beats (Gamma/Alpha/Theta/Delta)
   - Enable sidechain compression
   - Adjust export quality (192k/256k/320k)
   - Generate and test premium tracks

---

## 📊 Output Quality Comparison

| Metric | Free Version | Premium Version |
|--------|--------------|-----------------|
| **Sample Rate** | 44.1 kHz | 32 kHz (AI) → 44.1 kHz (export) |
| **Bit Depth** | 16-bit | 16-bit |
| **Channels** | Mono (1) | Stereo (2) |
| **Bitrate** | 128 kbps | 320 kbps |
| **Loudness** | Uncontrolled | -14 LUFS (broadcast standard) |
| **True Peak** | Uncontrolled | -1.5 dBTP (no clipping) |
| **Fades** | None | Professional (1.5s in, 3s out) |

---

## 🧪 Testing & Validation

### Test DSP Engine

```python
from premium.postprocess import inject_binaural_beats, apply_sidechain_compression
import numpy as np

# Generate test signal
sr = 32000
duration = 30
t = np.arange(sr * duration) / sr
audio = np.random.randn(sr * duration, 2) * 0.1  # Stereo noise

# Apply binaural beats
audio_binaural = inject_binaural_beats(audio, sr, base_freq=200, beat_freq=40)

# Apply sidechain
audio_sidechain = apply_sidechain_compression(audio, sr, bpm=128, strength=0.7)
```

### Validate Output Quality

```bash
# Analyze loudness
ffmpeg -i output.mp3 -af loudnorm=print_format=json -f null -

# Check bitrate
ffprobe -v error -show_entries format=bit_rate -of default=noprint_wrappers=1:nokey=1 output.mp3
```

---

## 💡 Tips for Best Results

### Prompt Engineering

✅ **Good Prompts:**
```
"ambient electronic focus music, minimal beats, warm analog pads, deep atmosphere, no vocals"
"uplifting progressive house, 126 BPM, emotional melodic leads, euphoric breakdown"
"deep meditation music, slowly evolving drones, healing frequencies, peaceful"
```

❌ **Poor Prompts:**
```
"good music" (too vague)
"fast loud techno with lots of bass and synths and drums" (too complex)
"happy birthday song" (copyrighted/specific songs don't work)
```

### Performance Optimization

1. **Batch Generation**: Generate all tracks at once to amortize model loading time
2. **Chunk Size**: Use 30s chunks for best quality/speed balance
3. **Model Selection**:
   - `stereo-large`: Best quality, 16GB VRAM, ~2-3 min/track
   - `stereo-medium`: Good quality, 8GB VRAM, ~1-2 min/track

---

## 🐛 Troubleshooting

### "CUDA out of memory"

```bash
# Use medium model instead
python -m premium.musicgen_daily \
  --model facebook/musicgen-stereo-medium \
  --date 2025-01-15
```

### "FFmpeg not found"

```bash
# Install FFmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# Colab:
!apt-get -qq install ffmpeg

# macOS:
brew install ffmpeg
```

### "audiocraft not installed"

```bash
pip install audiocraft
```

### Generation is slow

- Use Colab Pro for A100 GPU (10x faster than T4)
- Reduce `target_total_sec` in prompts
- Generate shorter tracks and loop them client-side

---

## 📚 Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PREMIUM PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Text Prompt                                             │
│       ↓                                                     │
│  2. MusicGen Stereo Large (3.3B params)                    │
│       ↓                                                     │
│  3. Raw Stereo Audio (32kHz, float32)                      │
│       ↓                                                     │
│  4. DSP Engine (Python/NumPy)                              │
│       ├─ Binaural Beat Injection (sine wave math)         │
│       ├─ Sidechain Compression (envelope shaping)         │
│       └─ Fade In/Out (sine-squared curves)                │
│       ↓                                                     │
│  5. FFmpeg Mastering                                       │
│       ├─ Loudness Normalization (-14 LUFS)                │
│       ├─ True Peak Limiting (-1.5 dBTP)                   │
│       └─ MP3 Encoding (320kbps, 44.1kHz)                  │
│       ↓                                                     │
│  6. Premium Track Output                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤝 Contributing

See main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

Premium-specific contributions:
- Additional DSP effects (chorus, delay, EQ)
- Better chunk stitching (crossfade)
- Real-time generation API
- Cloud deployment guides (AWS, GCP, Azure)

---

## 📄 License

MIT License - see [LICENSE](../LICENSE)

---

## 🙏 Credits

- **MusicGen**: Meta AI Research
- **Audiocraft**: Meta AI
- **DSP Algorithms**: Community contributions
- **SoundFlow**: Ruslan Magana Vsevolodovna

---

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/ruslanmv/soundflow/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ruslanmv/soundflow/discussions)
- 📧 **Email**: [Your contact email]

---

**Made with ❤️ by the SoundFlow team**

*Premium AI Music Production for Everyone*
