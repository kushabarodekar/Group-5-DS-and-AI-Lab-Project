# Kaggle GPU Fallback Deployment (Milestone 6)

Use this when **Lightning AI free-tier GPU/runtime quotas** may interrupt the live demo.
Kaggle notebook sessions with **GPU enabled** typically provide a longer stable window
(~9 hours) for final evaluation.

The fallback runs the **same integrated pipeline** as `app.py`:
`wellness_core.py` (5 models + risk fusion) + `report_dashboard.py` (Gradio UI).

---

## Prerequisites

1. **Kaggle account** with phone verification (required for GPU).
2. **Model weights** in `models/` (same files as local/Lightning deploy):
   - `Video_Fatigue.pth`
   - `Landmark_Fatigue.pt`
   - `Driver_Activity.pth`
   - `Smoking_And_Drinking.pt`
   - `SeatBelt_And_Phone.pt`
   - `m4_normalization_stats_ws45.csv`
3. Upload weights as a **Kaggle Dataset** (recommended) or commit via Git LFS if the
   repo is linked to Kaggle.

---

## Option A — One-command launcher (recommended)

1. Open [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**.
2. **Settings → Accelerator → GPU** (T4 or P100).
3. Add this repo as a **Kaggle Dataset** or clone from GitHub:

   ```python
   !git clone https://github.com/YOUR_ORG/Risk-Fusion-Engine-Final.git
   %cd Risk-Fusion-Engine-Final
   ```

4. Copy/link model weights into `models/` if using a separate dataset:

   ```python
   !cp -r /kaggle/input/your-weights-dataset/* models/
   ```

5. Install dependencies and run the fallback launcher:

   ```bash
   pip install -r requirements.txt
   python run_kaggle_gradio_fallback.py
   ```

6. Wait for the log line containing **`Running on public URL:`** — that is the
   temporary Gradio link to share with evaluators (valid while the notebook runs).

---

## Option B — Notebook entry point

Open `kaggle/Risk_Fusion_M6_Fallback.ipynb` in Kaggle (upload or import from repo)
and run all cells top-to-bottom. The last cell starts the same launcher.

---

## Verify GPU before demo

```python
!python scripts/check_cuda.py
```

Expected: `cuda available: True` and a GPU device name.

---

## Linux / MediaPipe note

Kaggle runs headless Linux. This repo bundles `native_libs/linux-x86_64/` for
MediaPipe (`libGLESv2`). The launcher sets `LD_LIBRARY_PATH` automatically.
If landmark fatigue fails to load:

```bash
pip install zstandard
python scripts/provision_gles.py
export LD_LIBRARY_PATH="${PWD}/native_libs/linux-x86_64:${LD_LIBRARY_PATH:-}"
python run_kaggle_gradio_fallback.py
```

---

## Lightning AI vs Kaggle

| | Lightning AI (free tier) | Kaggle GPU fallback |
|---|--------------------------|---------------------|
| GPU quota | Short / strict | ~9 h session typical |
| Public URL | Studio port / Gradio share | Gradio `share=True` |
| Entry point | `python app.py` | `python run_kaggle_gradio_fallback.py` |
| Inference core | `wellness_core.py` | **Same** |

**Do not remove Lightning deployment** — keep it as primary; use Kaggle as backup.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| No public URL | Ensure `share=True`; check Kaggle outbound network is allowed |
| `Device: cpu` | Enable GPU in notebook Settings; reinstall CUDA torch if needed |
| Models not found | Copy weights into `models/`; see `PLACE_WEIGHTS_HERE.md` |
| Session died | Re-run launcher cell; Gradio URL changes each session |
| Port in use | Restart kernel or set `server_port=7861` in launcher overrides |

---

## For evaluators (quick checklist)

1. Open the shared **Gradio public URL** from the running Kaggle notebook.
2. **Recorded video** tab — upload a short driving clip; wait for fused score + dashboard.
3. **Live webcam** tab — optional; requires browser camera permission.
4. Confirm startup log shows `Device: cuda` when GPU is enabled.
