# Prompt pour créer le notebook Colab AI-Scientist v1

Copiez-colle ce prompt dans un autre onglet/chat avec un assistant IA :

---

```
You are an expert AI programming assistant.

Create a complete Google Colab notebook for AI-Scientist v1 by SakanaAI (https://github.com/SakanaAI/AI-Scientist).

The notebook should be designed for Google Colab's free tier (T4 GPU, ~12GB RAM) and include:

## Notebook structure (9 cells):

### Cell 1 (markdown): Title
- Title: "🧪 AI-Scientist v1 on Google Colab"
- Subtitle: "Fully Automated Scientific Discovery — Powered by DeepSeek V4 Flash 🚀"
- Brief overview of the pipeline
- Warning: executes LLM-written code, use sandbox
- Cost: ~$0.50-3 per run with DeepSeek V4 Flash

### Cell 2 (markdown): Setup instructions
- Explain that the user needs API keys for at least one LLM provider
- Supported: OpenAI (OPENAI_API_KEY), Anthropic (ANTHROPIC_API_KEY), DeepSeek (DEEPSEEK_API_KEY)
- Mention they should add API keys using Colab's userdata module: `from google.colab import userdata`
- DeepSeek V4 Flash (model: `deepseek-v4-flash`) is ~10x cheaper than GPT-4o
- Get DeepSeek key at: https://platform.deepseek.com/api_keys

### Cell 3 (code): Environment setup
```python
import os, sys, json
from pathlib import Path
from google.colab import userdata

WORKING_DIR = Path("/content")
PROJECT_DIR = WORKING_DIR / "AI-Scientist"
os.chdir(WORKING_DIR)

# Clone repo
!git clone https://github.com/SakanaAI/AI-Scientist.git
%cd AI-Scientist

# Install LaTeX (smaller package for Colab)
!apt-get update -qq && apt-get install -y -qq texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra poppler-utils > /dev/null 2>&1

# Install Python packages
!pip install -q -r requirements.txt

# Verify GPU
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ No GPU detected")
```

### Cell 4 (code): Load API keys
```python
# Load API keys from Colab secrets
# Add them in: Runtime → Secrets → Add new secret
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY") or ""
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY") or ""
os.environ["DEEPSEEK_API_KEY"] = userdata.get("DEEPSEEK_API_KEY") or ""
os.environ["S2_API_KEY"] = userdata.get("S2_API_KEY") or ""

# Show which keys are set
for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "S2_API_KEY"]:
    status = "✅ Set" if os.environ.get(k) else "❌ Not set"
    print(f"   {status}: {k}")
```

### Cell 5 (code): Setup NanoGPT template (optional)
```python
%cd {PROJECT_DIR}

print("📦 Preparing NanoGPT data...")
!python data/enwik8/prepare.py
!python data/shakespeare_char/prepare.py
!python data/text8/prepare.py

print("🏃 Running baseline experiment (this takes 10-30 minutes)...")
%cd templates/nanoGPT
!python experiment.py --out_dir run_0
!python plot.py
%cd {PROJECT_DIR}
print("✅ NanoGPT baseline ready!")
```

### Cell 6 (markdown): How to launch
Explain the launch command:
- Use: `python launch_scientist.py --model "deepseek-v4-flash" --experiment nanoGPT_lite --num-ideas 2`
- Lite experiments: nanoGPT_lite, 2d_diffusion_lite, grokking_lite
- For 2D Diffusion: first install NPEET
- Recommended models:
  - `deepseek-v4-flash` (cheapest, ~$0.50-1/run)
  - `gpt-4o-mini` (good quality, ~$1-2/run)  
  - `claude-3-5-sonnet-20241022` (best quality, ~$15/run)
  - `deepseek-coder-v2-0724` (also works via DeepSeek API)

### Cell 7 (code): Launch AI Scientist
```python
%cd {PROJECT_DIR}

# ✏️ CONFIGURE YOUR RUN HERE
MODEL = "deepseek-v4-flash"      # LLM model for experiments
EXPERIMENT = "nanoGPT_lite"      # Template: nanoGPT_lite, 2d_diffusion_lite, grokking_lite
NUM_IDEAS = 2                    # Number of ideas to try
PARALLEL = False                 # Colab has 1 GPU, keep False

print("=" * 50)
print("🚀 Launching AI Scientist v1")
print("=" * 50)
print(f"   Model:      {MODEL}")
print(f"   Experiment: {EXPERIMENT}")
print(f"   Ideas:      {NUM_IDEAS}")
print()

!python launch_scientist.py \
    --model {MODEL} \
    --experiment {EXPERIMENT} \
    --num-ideas {NUM_IDEAS} \
    2>&1

print()
print("✅ AI Scientist run complete!")
```

### Cell 8 (code): Results summary
```python
%cd {PROJECT_DIR}

print("📂 Results:\n")

# Find PDFs
import glob
pdfs = sorted(glob.glob("**/*.pdf", recursive=True))
if pdfs:
    for pdf in pdfs:
        size = os.path.getsize(pdf) / 1e6
        print(f"   📄 {pdf} ({size:.1f} MB)")
else:
    print("   No PDFs found yet.")

# Find experiment folders
folders = sorted(glob.glob("templates/**/run_*", recursive=True))
if folders:
    print()
    for f in folders:
        print(f"   📁 {f}")

print("\n💡 Download: Files sidebar (📁 icon) → right-click → Download")
```

### Cell 9 (markdown): Tips
- Colab session disconnects after ~12h of inactivity
- Save checkpoints to Google Drive: mount Drive with `from google.colab import drive; drive.mount('/content/drive')` and copy results
- For 2D Diffusion template, first run:
  ```
  !git clone https://github.com/gregversteeg/NPEET.git
  %cd NPEET
  !pip install .
  %cd {PROJECT_DIR}
  ```
- DeepSeek V4 Flash (`deepseek-v4-flash`) is the most cost-effective option
- If you get CUDA OOM, use Lite experiment variants which are designed for lower memory

IMPORTANT:
- The notebook MUST work on Google Colab free tier (T4 GPU, ~12GB RAM)
- All cells must be self-contained and run sequentially
- Use `!` for shell commands in code cells
- Include error handling for missing API keys
- Use `from google.colab import userdata` for secrets (NOT kaggle_secrets)
- Do NOT use `--parallel` flag (Colab has only 1 GPU)
- Output should be a complete .ipynb file
```
