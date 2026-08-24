# Driver Wellness — Risk Fusion Engine on Lightning.ai

Step-by-step guide to run the **5-model Risk Fusion Engine** (Gradio app) on [Lightning.ai](https://lightning.ai) and share a public link with teammates, professors, or graders.

**Repo files involved:**
- `app.py` — Gradio UI (recorded video + live webcam)
- `wellness_core.py` — models + fusion logic
- `requirements.txt` — Python dependencies
- `models/` — checkpoint weights (you must add these)

**Time estimate:** ~30–45 min first time (mostly installing dependencies + uploading weights).

---

## 1. What you get at the end

- A **public web URL** (Gradio `*.gradio.live` link)
- Two tabs:
  - **Recorded video** — upload a driving clip → annotated video + fused risk score
  - **Live webcam** — real-time demo (slower; prefer recorded video for presentations)
- Five models fused with **Option A exponential risk fusion**

---

## 2. Prerequisites

### 2.1 Accounts
- Free [Lightning.ai](https://lightning.ai) account
- (Recommended) Hugging Face account if weights are stored in a HF Model repo

### 2.2 Model weights (required in `models/`)

| File | Module |
|------|--------|
| `Video_Fatigue.pth` | Video fatigue (EfficientNet-B0 + BiLSTM) |
| `Landmark_Fatigue.pt` | Landmark fatigue (MediaPipe + LSTM) |
| `Driver_Activity.pth` | Driver activity (MobileNetV3) |
| `Smoking_And_Drinking.pt` | Smoking / drinking (YOLOv8) |
| `SeatBelt_And_Phone.pt` | Seat belt & phone (YOLOv8) |
| `m4_normalization_stats_ws45.csv` | Landmark normalization stats |
| `face_landmarker.task` | MediaPipe face model (optional — auto-downloads if missing) |

> **Video Fatigue:** use the exported checkpoint from training (`Video_Fatigue.pth` = copy of `driver_wellness_best_macro_f1.pth`).

### 2.3 Hardware strategy (save free GPU hours)

| Step | Use |
|------|-----|
| Install packages, upload code/weights | **CPU** Studio (free) |
| Run the app | **GPU** Studio (T4 or L4) |

Lightning gives new users **~80 free GPU hours**. **Stop the GPU** when you are not demoing.

---

## 3. Create a Lightning Studio

1. Go to [https://lightning.ai](https://lightning.ai) and sign up / log in.
2. Open **[studio.lightning.ai](https://studio.lightning.ai)**.
3. Click **+ New Studio** (or **Create Studio**).
4. Name it: `driver-wellness-risk-fusion`.
5. Leave hardware on **CPU** for now → **Create**.
6. Wait until the Studio opens (browser IDE + terminal).

---

## 4. Upload the project

### Option A — Git clone (if the repo is on GitHub)

In the Studio **Terminal**:

```bash
cd /teamspace/studios/this_studio
git clone https://github.com/YOUR_ORG/Risk-Fusion-Engine.git
cd Risk-Fusion-Engine
```

### Option B — Upload a ZIP from your laptop

On your machine:

```bash
cd /path/to/Risk-Fusion-Engine
zip -r Risk-Fusion-Engine.zip . -x "*.venv*" -x "__pycache__/*"
```

In Lightning Studio: upload `Risk-Fusion-Engine.zip` to `/teamspace/studios/this_studio/`, then:

```bash
cd /teamspace/studios/this_studio
unzip Risk-Fusion-Engine.zip -d Risk-Fusion-Engine
cd Risk-Fusion-Engine
```

---

## 5. Add model weights

### Option A — Hugging Face Model repo (recommended for teams)

1. Create a HF **Model** repo (e.g. `your-team/driver-wellness-weights`).
2. Upload all files from the table in Section 2.2.
3. In Lightning terminal:

```bash
cd /teamspace/studios/this_studio/Risk-Fusion-Engine
pip install huggingface_hub
mkdir -p models

python - <<'PY'
from huggingface_hub import hf_hub_download

REPO = "YOUR_USERNAME/driver-wellness-weights"  # <-- CHANGE THIS

files = [
    "Video_Fatigue.pth",
    "Landmark_Fatigue.pt",
    "Driver_Activity.pth",
    "Smoking_And_Drinking.pt",
    "SeatBelt_And_Phone.pt",
    "m4_normalization_stats_ws45.csv",
    "face_landmarker.task",
]

for name in files:
    try:
        hf_hub_download(repo_id=REPO, filename=name, local_dir="models")
        print("OK:", name)
    except Exception as exc:
        print("SKIP:", name, "-", exc)
PY

ls -lh models/
```

### Option B — Manual upload

Use the Studio file browser to upload each weight file into:

```
/teamspace/studios/this_studio/Risk-Fusion-Engine/models/
```

Verify:

```bash
ls -la models/
```

You should see at least **5** large model files (`.pth` / `.pt`) plus the CSV.

---

## 6. Install dependencies (CPU Studio)

```bash
cd /teamspace/studios/this_studio/Risk-Fusion-Engine
pip install -r requirements.txt
```

This takes **5–10 minutes** the first time. Installs persist if you restart the same Studio.

---

## 7. Verify models load (optional, CPU or GPU)

```bash
cd /teamspace/studios/this_studio/Risk-Fusion-Engine
python -c "import wellness_core as c; c.build_manager(); print('All models loaded OK')"
```

If you see `Checkpoint not found`, fix `models/` before continuing.

---

## 8. Switch to GPU

1. In the Studio toolbar, open **Machine** / **Change machine**.
2. Select a **GPU** — **T4** or **L4** (16 GB VRAM recommended for 5 models).
3. Wait for the Studio to restart.
4. Confirm CUDA:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected: `CUDA: True` and a GPU name (e.g. `Tesla T4`).

---

## 9. Start the Gradio app

Run this from the project folder:

```bash
cd /teamspace/studios/this_studio/Risk-Fusion-Engine
python - <<'PY'
import app
app.demo.queue().launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    ssr_mode=False,
)
PY
```

**First startup:** terminal shows `Loading models...` for **1–3 minutes**. Do not stop the process.

When ready, you will see:

```text
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://xxxxxxxxxxxxx.gradio.live
```

**Share the `gradio.live` link** with your team. It stays valid for roughly **72 hours** while the app is running.

### Alternative: run `app.py` directly

If `app.py` was updated with:

```python
demo.queue().launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    ssr_mode=False,
)
```

then simply:

```bash
python app.py
```

### Alternative: Lightning Ports plugin

If `share=True` does not work:
1. Launch with `share=False` on port `7860`.
2. In Studio sidebar → **Ports** → expose port **7860**.
3. Open the URL Lightning provides.

---

## 10. How to use the app

### Recorded video tab (recommended for demos)

1. Open the public Gradio link.
2. Go to **Recorded video**.
3. Upload a **short clip (15–60 seconds)** — MP4 works best.
4. Click **Analyze**.
5. Wait for:
   - Annotated output video
   - Session summary (overall score, risk level, module contributions)

### Live webcam tab

1. Go to **Live webcam**.
2. Click **Start / Reset session**.
3. Allow browser camera access.
4. Annotated feed and live score update continuously.
5. Click **Finish & summarize** for the final summary.

> Live mode runs **all 5 models per frame** and is heavier than recorded video. Use a **short demo** or prefer recorded video for graded presentations.

---

## 11. Risk fusion (what the score means)

The app combines five module outputs using **Common Driver Risk Score Framework — Option A**:

```
R_i           = severity_weight × confidence   (safe predictions → 0)
R_total       = sum of R_i over active modules
overall_score = 100 × (1 − exp(−k × R_total))   with k = 0.05
```

Scores may be smoothed over a short window during streaming.

---

## 12. Save GPU credits

| Do this | When |
|---------|------|
| Use **CPU** Studio | Installing packages, editing code |
| Use **GPU** Studio | Only while running the app |
| **Stop** or sleep the Studio | After your demo / end of day |
| Keep videos **under 60 s** | Faster runs, less GPU time |

Check remaining credits: Lightning dashboard → **Billing**.

---

## 13. Troubleshooting

| Problem | What to do |
|---------|------------|
| `Checkpoint not found` | Run `ls models/` — all 5 weight files must be present |
| `CUDA out of memory` | Use a shorter video; switch to **L4** GPU; restart Studio |
| `Loading models...` hangs | Wait 3–5 min; check logs for missing file |
| Public link not working | Use `server_name="0.0.0.0"`, port `7860`, `share=True` |
| App very slow | Confirm GPU: `nvidia-smi` in terminal |
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` |
| Studio disconnected | Re-open Studio, switch to GPU, re-run Section 9 |

---

## 14. Quick reference (copy-paste)

```bash
# --- Setup (CPU) ---
cd /teamspace/studios/this_studio
git clone https://github.com/YOUR_ORG/Risk-Fusion-Engine.git
cd Risk-Fusion-Engine
pip install -r requirements.txt
# Add weights to models/ (HF download or manual upload)

# --- GPU check (after switching to GPU in UI) ---
python -c "import torch; print(torch.cuda.is_available())"

# --- Run app ---
python - <<'PY'
import app
app.demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True, ssr_mode=False)
PY
```

---

## 15. Other deployment options (if Lightning limits are hit)

| Platform | Best for |
|----------|----------|
| **Hugging Face Spaces (ZeroGPU)** | Free public app; GPU burst on "Analyze" only |
| **Google Colab + Gradio share** | One-off presentation (~12 h session) |
| **Kaggle + Gradio** | Same as training environment; 12 h session limit |
| **Local laptop + `share=True`** | Fastest if someone has an NVIDIA GPU |

See `README.md` for Hugging Face Space deployment notes.

---

## 16. Support checklist before asking for help

- [ ] All 5 model files in `models/`
- [ ] `pip install -r requirements.txt` completed without errors
- [ ] Studio is on **GPU** (not CPU) when running the app
- [ ] Terminal shows `Running on public URL: https://....gradio.live`
- [ ] Test clip is **≤ 60 seconds**

---

**Document version:** August 2026  
**Project:** Driver Wellness AI — Risk Fusion Engine  
**Maintainer:** DS/AI Lab Group 5
