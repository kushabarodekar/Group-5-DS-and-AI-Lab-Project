# AI-Powered Driver Wellness & Safety Monitoring System
# Developer Guide & Code Documentation — M1–M5

**Primary implementation artifact:** `Driver_Wellness_AI_Integrated_Updated_Live.ipynb`  
**Basis:** supplied integrated notebook, consolidated M1–M5 technical reports, milestone reports and deployment notes.

## 1. Purpose

This guide enables another developer to reproduce, execute, debug and extend the integrated Driver Wellness AI implementation. The supplied notebook is the source of truth for the current runtime configuration.

The system integrates five modules:

1. Video Fatigue Detection
2. Landmark Fatigue Detection
3. Driver Activity Recognition
4. Smoking & Drinking Detection
5. Seat Belt & Phone Usage Detection

All module adapters return a common `PredictionResult`, which is consumed by the Risk Fusion Engine and the shared streaming orchestrator.

## 2. Environment

- Google Colab
- Python 3.11
- GPU recommended; project evaluation used NVIDIA Tesla T4
- PyTorch / torchvision
- OpenCV
- NumPy
- Pillow
- Pandas
- Matplotlib
- Ultralytics
- MediaPipe
- psutil

The notebook explicitly installs:

```bash
%pip -q install "ultralytics>=8.3.0" "mediapipe>=0.10.14" psutil
```

The recorded run reported Ultralytics 8.4.116 and MediaPipe 1.0.0.

## 3. Checkpoints

| Module | Checkpoint |
|---|---|
| Video Fatigue | `Video_Fatigue.pth` |
| Landmark Fatigue | `Landmark_Fatigue.pt` |
| Driver Activity | `Driver_Activity.pth` |
| Smoking & Drinking | `Smoking_And_Drinking.pt` |
| Seat Belt & Phone | `SeatBelt_And_Phone.pt` |
| Landmark normalization | `m4_normalization_stats_ws45.csv` |
| MediaPipe asset | `face_landmarker.task` |

The registry should report **5/5 Available** before a complete integrated run.

## 4. First Run

1. Open `Driver_Wellness_AI_Integrated_Updated_Live.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Run dependency installation.
4. Mount Google Drive.
5. Configure the model root.
6. Run Model Registry.
7. Verify all five checkpoints.
8. Register adapters and warm up models.
9. Choose `video` or `live`.
10. Run the shared streaming pipeline.
11. Inspect module outputs.
12. Inspect risk fusion and dashboard.

Device selection is automatic:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 5. Logical Architecture

```text
Input Video / Webcam
        |
        v
StreamingWellnessOrchestrator
        |
        +--> VideoFatigueAdapter
        +--> LandmarkFatigueAdapter
        +--> DriverActivityAdapter
        +--> SmokingAdapter
        +--> SeatBeltPhoneDetectionAdapter
        |
        v
PredictionResult
        |
        v
DriverWellnessRiskFusion
        |
        +--> Overall risk
        +--> Risk level
        +--> Module contributions
        |
        v
Matplotlib Dashboard + Session Summary
```

## 6. Core SDK

The notebook implements:

- `PredictionResult`
- `InferenceContext`
- `BaseModelAdapter`
- `AdapterRegistry`
- `DriverWellnessModuleManager`
- standardized exceptions for missing checkpoints, invalid input, decode failures, buffer failures, face failures, unloaded models and GPU OOM.

The key design principle is that fusion and dashboard code consume standardized outputs rather than model-specific internals.

## 7. Module Implementations

### 7.1 Video Fatigue

- EfficientNet-B0 + BiLSTM
- Sequence length: 16
- Input: 224×224
- BiLSTM hidden size: 256
- 1 layer, bidirectional
- Dropout: 0.3
- Classes: Low Risk / Medium Risk / High Risk
- Risk mapping: 0.2 / 0.6 / 1.0
- Streaming stride: 4
- M5 test accuracy: 33.55%
- Reported latency: 21.03 ms/window

### 7.2 Landmark Fatigue

- MediaPipe Face Landmarker
- Features: EAR, MAR, Pitch, Yaw, Roll
- Window: 45 frames
- 2-layer LSTM
- Hidden size: 128
- Classes: Normal / Talking / Yawning
- Yawning threshold: 0.15
- M5 accuracy: ~64.07%
- Approximate model latency: ~0.20 ms/window, excluding MediaPipe extraction

### 7.3 Driver Activity

- MobileNetV3-Large
- Input: 224×224
- 5 classes
- Frame skip: 5
- Rolling probability history
- M5 test accuracy: 93.14%
- Parameters: ~4.2M
- Latency: ~12.5 ms

### 7.4 Smoking & Drinking

- YOLOv8n
- 640×640
- Global confidence: 0.25
- TemporalVoter
- Window: 0.50 s
- ON fraction: 0.50
- OFF fraction: 0.30
- Class floors: smoking 0.25, drinking 0.40
- Held-out test mAP@50: 0.8197
- Parameters: 3,011,238
- ~8.1 GFLOPs @ 640×640
- ~6.9 ms/frame on T4

### 7.5 Seat Belt & Phone

- YOLOv8n
- `SeatBelt_And_Phone.pt`
- 640×640
- Classes: Phone, Seatbelt
- Raw YOLO: conf 0.10, IoU 0.40
- Adapter floors: Phone 0.15, Seatbelt 0.20
- Temporal window: 0.50 s
- ON fraction: 0.30
- OFF fraction: 0.20
- Phone confirm: 3 sampled frames
- Phone release: 4 absence frames
- Seatbelt grace: 3 frames
- Current frame stride: 2
- Risk map:
  - Phone & Seatbelt: 0.85
  - Phone Only: 1.0
  - Seatbelt Only: 0.0
  - No Detection: 0.45
- Notebook-validation mAP@50: 0.9526
- Notebook-validation precision: 0.9370
- Notebook-validation recall: 0.9025
- Reported latency: 6.8–10.4 ms/frame
- ~3.2–3.5M parameters

**Important:** 0.9526 is notebook-validation performance, not a held-out test score.

## 8. Risk Fusion

Current integrated weights:

| Module | Weight |
|---|---:|
| Video Fatigue | 0.20 |
| Landmark Fatigue | 0.30 |
| Driver Activity | 0.20 |
| Smoking / Drinking | 0.15 |
| Seat Belt / Phone | 0.15 |

Risk bands:

- 0–25: Low
- >25–50: Moderate
- >50–75: High
- >75–100: Critical

## 9. Streaming

`StreamingWellnessOrchestrator`:

- decodes video once;
- maintains bounded temporal state;
- uses module-specific strides;
- reuses recent predictions between inference frames where applicable;
- fuses continuously at 1.0 s;
- summarizes every 15 s;
- throttles display to 2 FPS;
- supports recorded video and webcam;
- resets/finalizes state per session.

## 10. Input Modes

### Recorded

```python
INPUT_MODE = "video"
```

Supported: `.mp4`, `.avi`, `.mov`, `.mkv`, `.m4v`.

### Live

```python
INPUT_MODE = "live"
```

Local webcam uses `cv2.VideoCapture(0)`. Colab uses browser `getUserMedia`.

## 11. Notebook Code Inventory

| Cell | Responsibility |
|---:|---|
| 2 | Dependency installation |
| 4 | Google Drive mount |
| 6 | Configuration |
| 8 | Model registry |
| 10 | Input selection |
| 12 | Core SDK |
| 14 | Video preprocessing |
| 15 | Video Fatigue adapter |
| 17 | Driver Activity adapter |
| 19 | Landmark extraction |
| 20 | Landmark Fatigue adapter |
| 22 | Smoking adapter |
| 24 | Seat Belt/Phone adapter |
| 26 | Checkpoint loading |
| 28 | Adapter registration |
| 30 | Risk Fusion |
| 32 | Streaming orchestrator |
| 34 | Dashboard |
| 36 | Recorded video |
| 38 | Live webcam |
| 40 | Regression tests |

The notebook contains 42 total cells and is the complete supplied executable integration artifact.

## 12. Regression Tests

The notebook includes:

- `_test_temporal_voter_preserved`
- `_test_bounded_video_buffer`
- `_test_bounded_landmark_buffer`
- `_test_continuous_fusion_before_end`
- `_test_display_configured`
- `_test_input_mode_exclusive`

Run these after modifying temporal, buffering, fusion or input-mode logic.

## 13. Performance Baseline

| Module | Model | Metric | Latency |
|---|---|---|---|
| Video Fatigue | EfficientNet-B0 + BiLSTM | 33.55% test accuracy | 21.03 ms/window |
| Landmark | 2-layer LSTM | 64.07% accuracy | ~0.20 ms model inference* |
| Driver Activity | MobileNetV3-Large | 93.14% test accuracy | 12.5 ms |
| Seat Belt/Phone | YOLOv8n | 0.9526 mAP@50 validation | 6.8–10.4 ms/frame |
| Smoking/Drinking | YOLOv8n | 0.8197 mAP@50 held-out test | ~6.9 ms/frame |

\* excludes MediaPipe feature extraction.

## 14. YOLOv8n Engineering Decision

YOLOv8n should be described as the **validated and integrated project choice**, not as universally superior to YOLO11n or YOLOv8s.

The project evidence supports:

1. YOLOv8n was selected for real-time/edge-oriented operation.
2. The integrated Seat Belt/Phone module measured ~3.2–3.5M parameters and 6.8–10.4 ms/frame.
3. The finalized Smoking/Drinking model has 3,011,238 parameters and ~6.9 ms/frame on T4.
4. YOLOv8s is substantially larger in the referenced architecture specification.
5. The supplied project does not contain a controlled YOLOv8n vs YOLO11n vs YOLOv8s experiment on the same Seat Belt/Phone split.

## 15. Debugging

| Problem | Action |
|---|---|
| Missing checkpoint | Check `MODEL_ROOT`, filenames and registry |
| Unsupported video | Use `.mp4/.avi/.mov/.mkv/.m4v` |
| Face failure | Improve face visibility/lighting |
| GPU OOM | Shorten video / reduce inference frequency |
| YOLO CUDA issue | Inspect adapter device handling before changing device |
| Detection flicker | Use temporal voter/consensus |
| Long video slow | Increase stride gradually; reuse recent predictions |

## 16. Extension Rules

When adding a model:

1. Implement `BaseModelAdapter`.
2. Return `PredictionResult`.
3. Register through `AdapterRegistry`.
4. Add risk mapping and validated fusion weight.
5. Implement streaming reset/finalization if stateful.
6. Add regression tests.
7. Benchmark before/after.
8. Update the technical report and user guide.

## 17. Deployment Handoff

The deployment notes describe a Lightning ai + Gradio path with:

- `app.py`
- `wellness_core.py`
- `requirements.txt`
- `README.md`
- `.gitattributes`
- Git LFS model storage

Those wrapper-file sources are not present in the current supplied source bundle, so this guide does not invent their contents. The integrated notebook remains the complete supplied M1–M5 code artifact.

## 18. Final Developer Checklist

- [ ] Open notebook in Colab
- [ ] Select GPU
- [ ] Install dependencies
- [ ] Mount Drive
- [ ] Place all checkpoints/assets
- [ ] Verify 5/5 registry availability
- [ ] Register and warm up adapters
- [ ] Run short recorded video
- [ ] Inspect module outputs
- [ ] Inspect fusion/dashboard
- [ ] Run regression tests after code changes
- [ ] Record dependency/model/configuration changes
- [ ] Use actual M6 deployment repository for deployment wrappers

## Appendix — Code Artifact

The complete supplied notebook is included separately with this guide:

`Driver_Wellness_AI_Integrated_Updated_Live.ipynb`

The notebook should be version-controlled as the primary executable source rather than copying its contents into multiple documents.


---

# 19. M6 Final Deployment & Documentation Update

> **M6 source-of-truth update:** Existing M1–M5 developer-guide content is preserved. This section adds the deployment/documentation details supported by the corrected Milestone 6 Team Contribution Tracker.

## 19.1 Deployment Platform

The M6 tracker records **Lightning ai + Gradio** as the deployment approach.

Deployment has been initiated, with the following documented contributions:

- **Ravina:** prepared the initial deployment documentation and deployment guidance.
- **Shubham, Kushagra:** subsequently modified deployment parameters/configuration in Lightning ai platform and supported deployment validation.
- **Shiwani, Sohini:** attempted deployment using the similar technical approach based on Ravina's initial deployment.
- **All:** supported module-level deployment validation, integration testing and troubleshooting.

The current deployment status documented for M6 is that runtime inference is affected by **ZeroGPU quota-exceeded issues**. Therefore, deployment stabilization and further runtime testing remain part of the M6 deployment work.

## 19.2 M6 Integrated Application Structure

The final integrated application is organized around:

```text
Driver Video
      ↓
Gradio Interface
      ↓
Integrated Inference Pipeline
      ↓
┌───────────────────────────────────┐
│ Video Fatigue                     │
│ Landmark Fatigue                  │
│ Driver Activity                   │
│ Smoking & Drinking                │
│ Seat Belt & Phone                 │
└───────────────────────────────────┘
      ↓
Risk Fusion / Driver Wellness Score
      ↓
Risk Category
      ↓
Driver Safety Report / Annotated Output
```

The deployment package includes the core application files documented for M6:

```text
app.py
wellness_core.py
requirements.txt
README.md
models/
```

The exact checkpoint filenames and supporting files should remain aligned with the actual model files used by the project.

## 19.3 Gradio Interface

The M6 interface is intended to provide:

### Input

- Driver video upload
- Input validation
- Analysis control

### Output

- Annotated driver video
- Module-wise predictions
- Confidence scores
- Module risk scores
- Overall Driver Wellness Score
- Overall risk category
- Major safety warnings

The interface is intended to support the final integrated pipeline rather than individual standalone model demonstrations.

## 19.4 Lightning ai Deployment Workflow

The documented deployment workflow is:

```text
Prepare integrated code
        ↓
Prepare requirements.txt
        ↓
Prepare model checkpoints/configuration
        ↓
Create/configure lightning ai platform
        ↓
Upload/push application files
        ↓
Configure runtime/dependencies
        ↓
Launch Gradio application
        ↓
Test uploaded driver videos
        ↓
Validate module outputs
        ↓
Validate Risk Fusion output
```

Deployment testing should verify:

- Application starts successfully.
- Required dependencies are available.
- Model checkpoints can be loaded.
- Uploaded videos are accepted.
- Inference runs successfully.
- Annotated output is generated.
- Module results are displayed.
- Risk Fusion output is displayed.

## 19.5 Current Deployment Limitation

The corrected M6 tracker explicitly records a **ZeroGPU quota-exceeded issue during runtime inference**.

Therefore:

- Successful code preparation does not imply uninterrupted remote inference.
- Runtime deployment testing may be limited by available ZeroGPU quota.
- Further team support is required for stabilization and repeated end-to-end testing.
- Deployment limitations should be reported transparently in the final documentation.

## 19.6 Deployment Troubleshooting

| Issue | Recommended developer action |
|---|---|
| Missing dependency | Reinstall dependencies from `requirements.txt` |
| Model checkpoint not found | Verify checkpoint path and repository/Space files |
| Application fails at startup | Inspect Space build/runtime logs |
| Video processing fails | Check input format, codec and temporary-file handling |
| GPU/ZeroGPU runtime unavailable | Check current Space hardware/quota availability |
| ZeroGPU quota exceeded | Wait for quota availability or use an alternative runtime/deployment environment for testing |
| Slow inference | Test with short representative videos and profile the individual modules |
| One module fails | Isolate the module and verify its checkpoint, preprocessing and output contract |
| Risk Fusion output missing | Verify that all module outputs reach the fusion layer in the expected format |

## 19.7 M6 Developer Validation Checklist

- [ ] Five model modules integrated
- [ ] Common input/output format verified
- [ ] Risk Fusion connected
- [ ] Driver Wellness Score generated
- [ ] Risk category generated
- [ ] Gradio video upload implemented
- [ ] Annotated output implemented
- [ ] Module-wise results displayed
- [ ] Local application tested
- [ ] `app.py` uploaded/configured
- [ ] `wellness_core.py` uploaded/configured
- [ ] `requirements.txt` configured
- [ ] Model files/configuration available
- [ ] Remote inference tested where quota permits
- [ ] ZeroGPU quota limitation documented
- [ ] Deployment troubleshooting documented

## 19.8 Final M1–M6 Developer Workflow

```text
M1–M2  Problem Definition + Data
   ↓
M3     Architecture + Module Development
   ↓
M4     Training + Checkpoints
   ↓
M5     Evaluation + Error Analysis
   ↓
M6     Five-Model Integration
   ↓
       Risk Fusion / Driver Wellness Score
   ↓
       Gradio Application
   ↓
       Deployment in Lightning ai
   ↓
       Deployment Testing + Documentation
```

## 19.9 M6 Documentation Deliverables

The developer documentation supports the M6 deliverables:

- Integrated inference pipeline
- Individual model adapters/modules
- Risk Fusion Engine
- Gradio application
- Deployment configuration
- Dependencies
- Model/checkpoint requirements
- Deployment testing procedure
- Known deployment limitations
- Troubleshooting guidance

