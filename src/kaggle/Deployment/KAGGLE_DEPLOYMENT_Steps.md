# Driver Wellness — Risk Fusion Engine on Kaggle (GPU Fallback)

Step-by-step guide to run the **5-model Risk Fusion Engine** (Gradio app) on [Kaggle Notebooks](https://www.kaggle.com/code) when **Lightning AI free-tier GPU/runtime quotas** may interrupt the live demo.

This is the **backup deployment path**. Keep Lightning AI as primary; use Kaggle for a stable ~9-hour GPU session and a public Gradio link via `share=True`.

**Repo files involved:**
- `run_kaggle_gradio_fallback.py` — Kaggle launcher (`share=True`, GLES bootstrap)
- `app.py` — Gradio UI (recorded video + live webcam)
- `wellness_core.py` — models + fusion logic
- `requirements.txt` — Python dependencies
- `kaggle/Risk_Fusion_M6_Fallback.ipynb` — notebook entry point
- `models/` — checkpoint weights (you must add these)

**Time estimate:** ~30–45 min first time (dataset upload + pip install + first model load).

---

## 1. What you get at the end

- A **temporary public web URL** (Gradio `*.gradio.live` link from `share=True`)
- The **same integrated app** as Lightning (`app.py` → `wellness_core.py` + `report_dashboard.py`)
- Two tabs:
  - **Recorded video** — upload a driving clip → annotated video + fused risk score
  - **Live webcam** — real-time demo (heavier; prefer recorded video for presentations)
- Five models fused with **Option A exponential risk fusion**

> **Important:** The public URL is valid **only while the notebook cell is running**. If the kernel stops, you must re-run the launcher and share the new URL.

---

## 2. Prerequisites

### 2.1 Accounts
- Free [Kaggle](https://www.kaggle.com) account
- **Phone verification** on Kaggle (required before GPU can be enabled)
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

### 2.3 When to use Kaggle vs Lightning AI

| | Lightning AI (primary) | Kaggle (fallback) |
|---|------------------------|-------------------|
| Role | Default hosted demo | Backup when Lightning quota/runtime limits hit |
| GPU session | Short / strict free tier | ~9 h typical notebook session |
| Entry point | `python app.py` | `python run_kaggle_gradio_fallback.py` |
| Public URL | Gradio `share=True` → `gradio.live` | **Same** — `share=True` |
| Inference core | `wellness_core.py` | **Same** |

Kaggle gives **~30 GPU hours per week** (policy may change — check Kaggle → Account → GPU quota).

---

## 3. Create a Kaggle notebook

1. Go to [https://www.kaggle.com/code](https://www.kaggle.com/code) and sign up / log in.
2. Complete **phone verification** (Profile → Phone verification) if GPU is greyed out.
3. Click **+ New Notebook**.
4. Name it: `driver-wellness-risk-fusion-m6`.
5. Open **Settings** (right panel):
   - **Accelerator → GPU** (T4 or P100)
   - **Internet → On** (required for `pip install`, Gradio `share=True`, and MediaPipe downloads)
6. **Save** the notebook.

---

## 4. Upload the project

### Option A — Git clone inside the notebook (recommended)

In the first code cell:

```python
!git clone https://github.com/YOUR_ORG/Risk-Fusion-Engine-Final.git
%cd Risk-Fusion-Engine-Final
!ls -la
```

### Option B — Upload repo as a Kaggle Dataset

1. On your laptop, zip the project (exclude `.venv`, `__pycache__`):

```bash
cd /path/to/Risk-Fusion-Engine-Final
zip -r risk-fusion-engine.zip . -x "*.venv*" -x "__pycache__/*" -x ".git/*"
```

2. Kaggle → **Datasets → New Dataset** → upload `risk-fusion-engine.zip`.
3. In your notebook: **Add Input → Your Datasets** → select the dataset.
4. Copy into working directory:

```python
!cp -r /kaggle/input/risk-fusion-engine/* /kaggle/working/Risk-Fusion-Engine-Final/
%cd /kaggle/working/Risk-Fusion-Engine-Final
!ls -la
```

### Option C — Use the bundled notebook

Upload or import `kaggle/Risk_Fusion_M6_Fallback.ipynb` from the repo and run cells top-to-bottom.

---

## 5. Add model weights

Weights are **not** included in git by default. Use one of the following.

### Option A — Kaggle Dataset for weights (recommended for teams)

1. Create a Kaggle **Dataset** (e.g. `driver-wellness-weights`) with all files from Section 2.2.
2. In your notebook: **Add Input → Datasets** → attach `driver-wellness-weights`.
3. Copy into `models/`:

```python
%cd /kaggle/working/Risk-Fusion-Engine-Final   # adjust path if needed
!mkdir -p models
!cp -r /kaggle/input/driver-wellness-weights/* models/
!ls -lh models/
```

### Option B — Hugging Face Model repo

```python
%cd /kaggle/working/Risk-Fusion-Engine-Final
!pip install -q huggingface_hub
!mkdir -p models

import os
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

!ls -lh models/
```

### Option C — Manual upload to Kaggle Dataset

Upload each weight file through the Kaggle Dataset UI, then use Option A copy commands.

Verify:

```bash
ls -la models/
```

You should see at least **5** large model files (`.pth` / `.pt`) plus the CSV.

---

## 6. Install dependencies

In a notebook cell (GPU already enabled):

```python
%cd /kaggle/working/Risk-Fusion-Engine-Final
!pip install -q -r requirements.txt
```

First install takes **5–10 minutes**. Re-running the same notebook session reuses the environment until the kernel restarts.

Optional — ensure CUDA-enabled PyTorch (usually preinstalled on Kaggle GPU):

```python
!python scripts/check_cuda.py
```

Expected:

```text
cuda available: True
Tesla T4
```

---

## 7. Verify models load (optional)

```python
%cd /kaggle/working/Risk-Fusion-Engine-Final
!python -c "import wellness_core as c; c.build_manager(); print('All models loaded OK')"
```

If you see `Checkpoint not found`, fix `models/` before continuing.

---

## 8. Start the Gradio app (Kaggle launcher)

**Do not use `python app.py` for the Kaggle demo** — use the fallback launcher, which sets Kaggle-friendly defaults and bootstraps MediaPipe on headless Linux.

```python
%cd /kaggle/working/Risk-Fusion-Engine-Final
!python run_kaggle_gradio_fallback.py
```

What the launcher does:
- Detects Kaggle environment (`/kaggle/input`)
- Sets `LD_LIBRARY_PATH` for bundled `native_libs/linux-x86_64/` (MediaPipe `libGLESv2`)
- Calls `app.launch_app(share=True, server_name="0.0.0.0", server_port=7860)`
- Prints CUDA status before loading models

**First startup:** terminal shows `Loading models...` for **1–3 minutes**. **Do not stop the cell.**

When ready, you will see:

```text
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://xxxxxxxxxxxxx.gradio.live
```

**Copy and share the `gradio.live` link** with evaluators. Keep this notebook cell **running** for the entire demo.

### What `share=True` does

Gradio creates a temporary public tunnel to your Kaggle notebook process. No port forwarding setup is needed on Kaggle (unlike Lightning Ports plugin). The URL **changes each time** you restart the launcher.

---

## 9. How to use the app

### Recorded video tab (recommended for demos)

1. Open the shared Gradio link.
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

## 10. Risk fusion (what the score means)

The app combines five module outputs using **Common Driver Risk Score Framework — Option A**:

```
R_i           = severity_weight × confidence   (safe predictions → 0)
R_total       = sum of R_i over active modules
overall_score = 100 × (1 − exp(−k × R_total))   with k = 0.05
```

Scores may be smoothed over a short window during streaming.

Video fatigue is **down-weighted** in M6 (`video_fatigue` weight 0.10, trust factor 0.50) but remains visible on the dashboard.

---

## 11. Save GPU quota & session time

| Do this | When |
|---------|------|
| Enable **GPU** only for the demo notebook | Before running the launcher |
| Enable **Internet** | Required for pip, Gradio share, downloads |
| Keep videos **under 60 s** | Faster runs, less GPU time |
| **Stop** the notebook when done | Frees GPU for other users / saves weekly quota |
| Do not restart the launcher unnecessarily | Each restart = new public URL + model reload |

Check remaining GPU time: Kaggle → **Account** → **GPU** / quota dashboard.

Typical Kaggle GPU notebook session: **~9 hours** before auto-shutdown (policy may vary).

---

## 12. Troubleshooting

| Problem | What to do |
|---------|------------|
| GPU option greyed out | Complete Kaggle phone verification |
| `cuda available: False` | Settings → Accelerator → **GPU**; restart kernel |
| `Checkpoint not found` | Run `!ls models/` — all 5 weight files must be present |
| No public URL | Ensure launcher uses `share=True`; Internet must be **On** |
| `Loading models...` hangs | Wait 3–5 min; check logs for missing file |
| `libGLESv2.so.2` / landmark fails | Re-run launcher (auto-bootstrap); or run `python scripts/provision_gles.py` |
| App very slow | Confirm GPU: `!python scripts/check_cuda.py` and `!nvidia-smi` |
| `ModuleNotFoundError` | Re-run `!pip install -r requirements.txt` |
| Session died / link dead | Re-run `run_kaggle_gradio_fallback.py`; share the **new** URL |
| Port in use | Restart kernel or edit launcher to use `server_port=7861` |
| Gradio link works then stops | Notebook cell was stopped or session timed out — re-run launcher |

### MediaPipe / headless Linux (landmark fatigue)

Kaggle is headless Linux. The repo bundles GLES libraries under `native_libs/linux-x86_64/`.
The Kaggle launcher sets `LD_LIBRARY_PATH` automatically.

If landmark still fails:

```python
!pip install -q zstandard
!python scripts/provision_gles.py
import os
os.environ["LD_LIBRARY_PATH"] = "/kaggle/working/Risk-Fusion-Engine-Final/native_libs/linux-x86_64:" + os.environ.get("LD_LIBRARY_PATH", "")
!python run_kaggle_gradio_fallback.py
```

---

## 13. Quick reference (copy-paste)

```python
# --- Clone (adjust URL) ---
!git clone https://github.com/YOUR_ORG/Risk-Fusion-Engine-Final.git
%cd Risk-Fusion-Engine-Final

# --- Weights from Kaggle dataset (adjust slug) ---
!mkdir -p models
!cp -r /kaggle/input/driver-wellness-weights/* models/
!ls -lh models/

# --- Install + GPU check ---
!pip install -q -r requirements.txt
!python scripts/check_cuda.py

# --- Run Kaggle fallback demo (keep cell running) ---
!python run_kaggle_gradio_fallback.py
# Copy: Running on public URL: https://....gradio.live
```

---

## 14. Lightning AI vs Kaggle (side-by-side)

| Step | Lightning AI | Kaggle fallback |
|------|--------------|-----------------|
| Create environment | New Studio | New Notebook + GPU |
| Project path | `/teamspace/studios/this_studio/...` | `/kaggle/working/...` |
| Weights | HF download or Studio upload | Kaggle Dataset → `models/` |
| Install deps | `pip install -r requirements.txt` | Same |
| Run command | `python app.py` | `python run_kaggle_gradio_fallback.py` |
| Public link | `share=True` → `gradio.live` | **Same** |
| Keep alive | Leave Studio + process running | **Keep notebook cell running** |

See `LIGHTNING_AI_DEPLOYMENT_Steps.md` for the primary Lightning path.

---

## 15. Support checklist before asking for help

- [ ] Kaggle **phone verification** complete
- [ ] Notebook **Accelerator → GPU** enabled
- [ ] **Internet → On** in notebook settings
- [ ] All 5 model files in `models/`
- [ ] `pip install -r requirements.txt` completed without errors
- [ ] `scripts/check_cuda.py` shows `cuda available: True`
- [ ] Terminal shows `Running on public URL: https://....gradio.live`
- [ ] Launcher cell is **still running** (not interrupted)
- [ ] Test clip is **≤ 60 seconds**

---

**Document version:** August 2026  
**Project:** Driver Wellness AI — Risk Fusion Engine  
**Maintainer:** DS/AI Lab Group 5  
**Related:** `docs/KAGGLE_FALLBACK.md`, `kaggle/Risk_Fusion_M6_Fallback.ipynb`, `LIGHTNING_AI_DEPLOYMENT_Steps.md`
