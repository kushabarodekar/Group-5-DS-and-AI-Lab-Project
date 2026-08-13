# MILESTONE 6 — FINAL CONSOLIDATED PROJECT REPORT

## AI-Powered Driver Wellness & Safety Monitoring System

**Final Milestone:** Milestone 6 — Integration, Validation, Deployment & Documentation  
**Submission Date:** 13 August 2026  
**Team Size:** 5 Members  
**Modules:** 5  
**Final Application:** Gradio-based Driver Wellness AI  
**Primary Input:** Recorded driver video / live webcam  
**Core Pipeline:** Video Input → Streaming Orchestrator → Five Model Adapters → Standardized Predictions → Risk Fusion Engine → Driver Wellness Score (0–100)

---

## 1. Executive Summary

Milestone 6 represents the final integration and consolidation stage of the **AI-Powered Driver Wellness & Safety Monitoring System**. The project combines five independently developed machine-learning modules into a common end-to-end driver-monitoring pipeline:

1. **Video-Based Fatigue Detection** — Kushagra Barodekar
2. **Landmark-Based Fatigue Detection** — Shiwani Tiwari
3. **Driver Activity Classification** — Shubham
4. **Smoking & Drinking Detection** — Ravina
5. **Seat Belt & Phone Usage Detection** — Sohini Sarkar

The five modules use different modeling strategies because they address different observable aspects of driver wellness and safety. The final system standardizes their outputs through module adapters and combines their risk contributions through a common Risk Fusion Engine.

Milestone 6 focused on:
- integrating all five trained checkpoints;
- implementing standardized module interfaces;
- validating recorded-video and streaming behavior;
- adding temporal stabilization where appropriate;
- verifying risk-score propagation and final driver-level fusion;
- building a Gradio user interface;
- preparing Hugging Face Spaces deployment;
- handling module failures gracefully;
- consolidating the technical documentation and final project evidence.

The integrated architecture successfully processes a common driver video and activates the five registered modules. A cross-camera integration test processed **1,760 frames** with all five modules active and no module errors; the observed overall Driver Wellness Score was **49.8/100 (Moderate Risk)**. This result demonstrates end-to-end pipeline operation and risk-fusion integration, but it should not be interpreted as a system-wide accuracy benchmark because the ground-truth fatigue state for that specific clip was not independently re-verified.

The final project is therefore **technically integrated and locally functional**, with the Hugging Face deployment prepared but runtime inference currently subject to a documented ZeroGPU quota limitation. The modules have different readiness levels: the Driver Activity and YOLO-based behavior modules show strong practical performance, while the fatigue modules require additional cross-subject and real-world validation before being treated as standalone safety decision systems.

---

# 2. Project Objectives

The final system was designed to provide a multi-signal assessment of driver wellness and safety from a common video stream.

### 2.1 Primary Objectives

- Detect behavioral and physiological indicators of unsafe driving.
- Combine spatial, temporal, facial-landmark and object-detection signals.
- Process recorded videos and support live-webcam operation.
- Produce standardized module-level predictions.
- Convert module predictions into interpretable risk contributions.
- Generate a bounded **0–100 Driver Wellness Score**.
- Provide risk levels:
  - **0–25:** Low Risk
  - **>25–50:** Moderate Risk
  - **>50–75:** High Risk
  - **>75–100:** Critical Risk
- Provide annotated output and a user-facing Gradio interface.
- Support deployment to Hugging Face Spaces.

---

# 3. Team Contributions and Module Ownership

| Member | Module | Primary Model | Main Milestone 6 Responsibility |
|---|---|---|---|
| **Kushagra Barodekar** | Video-Based Fatigue Detection | EfficientNet-B0 + BiLSTM | Final model evaluation, root-cause experiments, fatigue-module consolidation, deployment |
| **Shiwani Tiwari** | Landmark-Based Fatigue Detection | Landmark features + LSTM | Feature validation, adapter integration, temporal fatigue logic, failure handling, deployment |
| **Shubham** | Driver Activity Classification | MobileNetV3-Large | Adapter, streaming support, Gradio interface, deployment support |
| **Ravina** | Smoking & Drinking Detection | YOLOv8n | Object-detection lifecycle, final evaluation, deployment-oriented initiation |
| **Sohini Sarkar** | Seat Belt & Phone Usage Detection | YOLOv8n | Integration, temporal stabilization, risk fusion, deployment support, documentation consolidation |

---

# 4. Overall System Architecture

The final system follows a modular architecture so that each model can be developed, evaluated and maintained independently.

```text
                         DRIVER VIDEO
                              |
                              v
                  Streaming / Video Orchestrator
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
   Video Fatigue       Landmark Fatigue     Driver Activity
   EfficientNet +      MediaPipe + LSTM     MobileNetV3-Large
   BiLSTM
          |                   |                   |
          +-------------------+-------------------+
                              |
          +-------------------+-------------------+
          |                                       |
          v                                       v
 Smoking & Drinking                    Seatbelt & Phone
 YOLOv8n                              YOLOv8n
          |                                       |
          +-------------------+-------------------+
                              |
                              v
                 Standardized PredictionResult
                              |
                              v
                     Risk Fusion Engine
                              |
                              v
                Driver Wellness Score (0–100)
                              |
                              v
                  Low / Moderate / High / Critical
                              |
                              v
                     Gradio Dashboard
```

Each adapter exposes standardized information including module identity, prediction, confidence, risk contribution and status. This common contract allows the Risk Fusion Engine to remain independent of the internal architecture of individual models.

---

# 5. Five-Module System Overview

| Module | Input/Task | Final Architecture | Key Output |
|---|---|---|---|
| Video Fatigue | Short video sequence classification | EfficientNet-B0 + BiLSTM | Safe / Caution / High_Risk |
| Landmark Fatigue | Facial landmark temporal analysis | MediaPipe features + LSTM | Alert / Mild Fatigue / Drowsy |
| Driver Activity | Driver behavior classification | MobileNetV3-Large | 5 activity classes |
| Smoking & Drinking | Object detection | YOLOv8n | Smoking / Drinking |
| Seatbelt & Phone | Object detection | YOLOv8n | Phone / Seatbelt states |

The five modules intentionally provide complementary evidence. Fatigue modules focus on physiological/temporal cues, the activity classifier models driving behavior, and the two YOLO modules detect visible unsafe behaviors.

---

# 6. Module 1 — Video-Based Fatigue Detection

## 6.1 Owner

**Kushagra Barodekar**

## 6.2 Final Architecture

- **Backbone:** ImageNet-pretrained EfficientNet-B0
- **Temporal model:** 1-layer bidirectional LSTM
- **LSTM hidden size:** 256
- **Dropout:** 0.30
- **Frame feature size:** 1280
- **Sequence length:** 16 frames
- **Sampling:** approximately 5 FPS
- **Temporal coverage:** approximately 3.2 seconds
- **Checkpoint:** `Video_Fatigue.pth`
- **Parameters:** approximately 7.16M

The EfficientNet backbone extracts frame-level visual representations and the BiLSTM models temporal information across the sequence.

## 6.3 Dataset and Processing

The module uses the **UTA Real-Life Drowsiness Dataset (UTA-RLDD)**.

The final audited index contained:
- **60 subjects**
- **60,244 indexed sequences**
- subject-disjoint train/validation/test split;
- approximately 70% / 15% / 15% subject allocation.

The final indexed sequence counts were:
- Train: 46,085
- Validation: 7,042
- Test: 7,117

Each sequence uses 16 frames at approximately 5 FPS. Frames are resized to 224×224 and converted from BGR to RGB.

Training preprocessing:
`ToTensor → ColorJitter → ImageNet Normalize → RandomErasing`

Validation/test preprocessing:
`ToTensor → ImageNet Normalize`

## 6.4 Final Evaluation

| Metric | Validation | Locked Test |
|---|---:|---:|
| Accuracy | **45.37%** | **32.86%** |
| Balanced Accuracy | **45.27%** | **33.22%** |
| Macro F1 | **44.31%** | **33.78%** |
| Weighted F1 | 44.37% | 33.47% |
| Minimum Class Recall | 32.36% | 25.84% |

The largest test failure was **Safe ↔ High_Risk confusion**, demonstrating a substantial unseen-subject generalization gap.

A controlled M6 sanity-overfit test reached **100% training accuracy**, showing that the pipeline can learn a small balanced subset. However, this does not establish generalization.

M6 controlled experiments found:
- partial backbone unfreezing improved accuracy on the diagnostic split but collapsed High_Risk recall;
- weighted cross-entropy produced almost no meaningful improvement;
- landmark fusion improved diagnostic Macro F1 but also collapsed High_Risk recall;
- landmark fusion plus partial unfreezing produced the highest diagnostic Macro F1 but was not sufficiently balanced for production use.

## 6.5 Readiness

The module is **technically functional but not standalone deployment-ready**. Its most appropriate current role is as a complementary fatigue signal inside the multi-module system.

Priority future work:
- broader subject-level validation;
- improved temporal context;
- better label/window alignment;
- confidence calibration;
- larger-scale landmark-fusion validation;
- realistic in-cabin robustness testing.

---

# 7. Module 2 — Landmark-Based Fatigue Detection

## 7.1 Owner

**Shiwani Tiwari**

## 7.2 Feature Pipeline

The module uses MediaPipe Face Landmarker to extract facial landmarks and a facial transformation matrix.

Five per-frame features are generated:

- **EAR** — Eye Aspect Ratio
- **MAR** — Mouth Aspect Ratio
- **Pitch**
- **Yaw**
- **Roll**

These features are accumulated into a **45-frame sliding window**, matching the training configuration. Each completed window is normalized using the saved training mean/std statistics before being passed to the LSTM classifier.

## 7.3 M6 Integration Corrections

Two important train/inference consistency issues were identified and corrected:

1. MAR landmark indices were restored to the same indices used during training.
2. Head-pose calculation was restored to the original MediaPipe transformation-matrix decomposition rather than the different `solvePnP` implementation.

The normalization-statistics loader also contained a column-name mismatch that could cause identity normalization. This was corrected and the true training statistics were verified.

## 7.4 Adapter and Temporal Logic

The module is integrated through:

`LandmarkFatigueAdapter`

with module key:

`landmark_fatigue`

The output follows the standardized contract:

```json
{
  "module": "landmark_fatigue",
  "prediction": "...",
  "confidence": 0.00,
  "risk_score": 0.00,
  "status": "OK"
}
```

A rolling history of the most recent 10 window-level predictions is used to derive fatigue state.

- `yawn_proportion = 0` → Alert
- `0 < yawn_proportion < threshold` → Mild Fatigue
- `yawn_proportion >= threshold` → Drowsy

The M6 threshold was set to **5%**, based on the documented validation sweep and the role of the module as one weighted signal among five.

A reliability gate prevents unsupported fatigue-state claims when the relevant Yawning-class F1 does not meet the required minimum.

## 7.5 Integration Testing

A cross-camera test used `11-MaleGlasses.avi`.

Results:
- Frames processed: **1,760**
- Elapsed time: **106.7 seconds**
- Active modules: all five
- Module errors: **None**
- Landmark prediction: **Drowsy**
- Landmark confidence: **0.831**
- Landmark status: **OK**
- Overall Driver Wellness Score: **49.8/100 — Moderate Risk**

This is evidence of end-to-end pipeline operation and cross-camera integration, not an independently validated accuracy result.

A separate unsuitable-camera test correctly returned a graceful unavailable/error state instead of crashing.

Live-webcam tests showed approximately **1.3–1.5 FPS**, with landmark extraction being the likely throughput bottleneck.

## 7.6 Limitations

- The deployed checkpoint is a hidden-size-128 variant, while the Milestone 5 evaluation used a hidden-size-32 checkpoint. The original checkpoint could not be located and the discrepancy is documented.
- The model is dependent on a reasonably front-facing driver view.
- Live webcam throughput is low.
- Broader real-world validation remains necessary.

---

# 8. Module 3 — Driver Activity Classification

## 8.1 Owner

**Shubham**

## 8.2 Model

The Driver Activity module uses **MobileNetV3-Large** with a five-class classifier.

Classes:
1. `other_activities`
2. `safe_driving`
3. `talking_phone`
4. `texting_phone`
5. `turning`

Input size:
- **224×224**

Streaming:
- **frame skip = 5**
- rolling probability history
- temporal smoothing through averaged probabilities.

Checkpoint:
`models/Driver_Activity.pth`

## 8.3 Adapter Integration

The module is registered as:

`DriverActivityAdapter`

under module key:

`driver_activity`

The adapter follows the common `BaseModelAdapter` interface and loads the MobileNetV3-Large checkpoint into evaluation mode.

Standardized output contains:
- prediction;
- confidence;
- risk score;
- predicted index;
- probability distribution;
- frames sampled;
- frame skip;
- checkpoint information.

## 8.4 Risk Mapping

Module-level risk mapping:

| Activity | Base Risk |
|---|---:|
| Safe driving | 0.05 |
| Turning | 0.20 |
| Talking phone | 0.60 |
| Other activities | 0.70 |
| Texting phone | 0.85 |

The module contributes to the common risk fusion system with a documented **20% fusion weight**.

## 8.5 Integrated Performance

| Metric | Standalone | Integrated | Status |
|---|---:|---:|---|
| Test Accuracy | 93.14% | **93.14%** | Maintained |
| Inference Speed | 12.5 ms | **12.5 ms** | Maintained |
| FPS | 80 | **80** | Maintained |
| Risk Score | — | 0.05–0.85 | Correct |

Integration scenarios covering normal driving, phone usage, distracted driving, turning and multiple-risk situations were documented as passing.

Short videos were handled gracefully. Low-quality/occluded footage can reduce performance.

## 8.6 Gradio Interface Contribution

Shubham led the user-facing Gradio interface, including:
- project header and module-status chips;
- recorded-video upload;
- processed-video output;
- module-wise predictions;
- overall risk gauge;
- live webcam tab;
- session controls;
- session summary generation.

A custom cockpit-style HUD presentation was implemented around the wellness dashboard.

---

# 9. Module 4 — Smoking & Drinking Detection

## 9.1 Owner

**Ravina**

## 9.2 Final Model

- **Architecture:** YOLOv8n
- **Classes:** smoking, drinking
- **Input:** 640×640 RGB
- **Epochs:** 80
- **Optimizer:** AdamW
- **Initial learning rate:** 0.001
- **Schedule:** cosine
- **Batch size:** 16
- **Seed:** 42
- **Parameters:** 3,011,238
- **Checkpoint:** `yolov8n_best.pt`

The model was selected after comparing YOLOv8n, YOLO11n and YOLOv8s. YOLOv8n won on the strict mAP@50–95 metric while maintaining the lightweight computational envelope.

## 9.3 Dataset

The source was a Roboflow YOLOv8 dataset.

After cleaning, duplicate removal and balancing:
- **3,704 images**
- Train/Validation/Test: **2,963 / 370 / 371**
- target classes: smoking and drinking.

Perceptual hashing removed duplicate clusters before splitting to reduce leakage risk.

## 9.4 Held-Out Test Results

| Metric | Overall | Smoking | Drinking |
|---|---:|---:|---:|
| Precision | **0.8465** | 0.9514 | 0.7415 |
| Recall | **0.8003** | 0.9299 | 0.6707 |
| mAP@50 | **0.8197** | 0.9255 | 0.7139 |
| mAP@50–95 | **0.4468** | 0.5300 | 0.3636 |

The primary weakness is the drinking class, especially recall.

The model produced 325 successes and 46 failures in the documented image-level analysis. Only six of the 46 failures were true cross-class confusions; most were misses or low-confidence correct detections.

## 9.5 Computational Profile

- Parameters: approximately **3.0M**
- Compute: approximately **8.1 GFLOPs at 640**
- Latency: approximately **6.9 ms/frame**
- Approximately **145 FPS on Tesla T4**

The model therefore has a strong real-time and edge-deployment profile.

## 9.6 Limitations

- Drinking detection is weaker than smoking detection.
- Small objects, especially cigarettes, create localization difficulty.
- Web-sourced training imagery does not fully represent real driver cabins.
- Night/infrared and unusual camera mounting conditions remain insufficiently validated.
- Single-frame inference can miss very brief actions.
- Fairness/per-group performance was not measured because demographic attributes were not available in the dataset.

---

# 10. Module 5 — Seat Belt & Phone Usage Detection

## 10.1 Owner

**Sohini Sarkar**

## 10.2 Final Model

- **Detector:** YOLOv8n
- **Classes:** Phone, Seatbelt
- **Parameters:** 3,011,238
- **Base configuration:** 640
- **Effective integrated inference:** 1280
- **Framework:** Ultralytics YOLO / PyTorch
- **Image/video processing:** OpenCV
- **Checkpoint:** final Seatbelt/Phone YOLOv8n checkpoint

The M6 integration adds video-oriented processing around the frame detector.

## 10.3 Robustness Pipeline

The final adapter uses:

1. 1280-pixel inference
2. Gamma shadow correction
3. Class-specific confidence floors
4. Spatial/geometric filtering
5. Class-agnostic NMS
6. Sliding-window temporal consensus
7. Hysteresis

Important configuration values:

| Setting | Value |
|---|---:|
| Effective inference size | 1280 |
| Raw YOLO confidence | 0.10 |
| Phone confidence floor | 0.02 |
| Seatbelt confidence floor | 0.35 |
| Temporal window | 0.50 s |
| Activation fraction | 0.40 |
| Release fraction | 0.20 |
| Frame stride | 5 |
| Gamma | 1.4 |

These measures address small phones, shadows, glare, reflections, driver-arm overlap and frame-to-frame flicker.

## 10.4 Module States

The module produces four stabilized states:

- **Phone & Seatbelt**
- **Phone Only**
- **Seatbelt Only**
- **No Detection**

The output is standardized through `PredictionResult` and passed to the common Risk Fusion Engine.

## 10.5 Documented Validation Results

| Metric | Value |
|---|---:|
| mAP@50 | **0.9526** |
| Precision | **0.9370** |
| Recall | **0.9025** |
| Parameters | 3,011,238 |
| Inference latency | ~6.8–10.4 ms/frame on documented T4 |
| VRAM | ~1.5–2.5 GB on documented T4 |

The reported detection metrics are validation-set values from the M5 project material, not an independently held-out M6 test benchmark.

## 10.6 Risk Fusion Contribution

The module uses the common event-severity × confidence formulation.

Examples of documented severity treatment include:
- Phone Only → severity 9
- Phone & Seatbelt → severity 9
- Seatbelt Only → safe contribution, severity 0
- No Detection → severity 8 in the current framework

The final driver score uses exponential fusion with:

`k = 0.05`

Optional fused-score smoothing uses a **3-second window**.

## 10.7 M6 Validation

Verified:
- checkpoint loading;
- class mapping;
- frame-level inference;
- bounding-box generation;
- four module-level states;
- confidence propagation;
- standardized output;
- Risk Fusion compatibility;
- integration into the common video pipeline.

Known limitations include heavy cabin shadows, strong sunlight/glare, reflections and arm/torso overlap.

---

# 11. Common Risk Fusion Framework

The system separates **model-specific prediction** from **model-agnostic driver-level risk fusion**.

For each risky event:

`R_i = severity_weight_i × confidence_i`

The total event risk is:

`R_total = Σ R_i`

The final score is converted to a bounded 0–100 scale using exponential fusion:

`Overall Score = 100 × (1 − exp(−0.05 × R_total))`

The resulting score is mapped to:

| Score | Risk Level |
|---:|---|
| 0–25 | Low Risk |
| >25–50 | Moderate Risk |
| >50–75 | High Risk |
| >75–100 | Critical Risk |

This architecture allows individual models to remain independently maintainable while providing a common final wellness signal.

---

# 12. End-to-End Integration

The M6 integration brings all five checkpoints into a single model manager and streaming orchestrator.

The common execution sequence is:

1. Driver video is uploaded or captured from webcam.
2. The streaming orchestrator decodes the video.
3. Frames are selectively processed according to module-specific requirements.
4. Each module adapter performs its own inference.
5. Temporal modules maintain their respective histories/windows.
6. Object detectors perform detection and stabilization.
7. Every module returns a standardized `PredictionResult`.
8. The Risk Fusion Engine converts predictions into event-level risk.
9. Module risks are combined into the Driver Wellness Score.
10. The Gradio interface displays annotated output, predictions, risk level and session information.

The five-module registration is:

| Module Key | Module |
|---|---|
| `video_fatigue` | Video-Based Fatigue |
| `landmark_fatigue` | Landmark-Based Fatigue |
| `driver_activity` | Driver Activity |
| `smoking` | Smoking & Drinking |
| `seatbelt` | Seat Belt & Phone |

---

# 13. Integrated System Testing

A documented cross-camera integration test used `11-MaleGlasses.avi`.

| Test Property | Result |
|---|---|
| Frames processed | 1,760 |
| Elapsed time | 106.7 s |
| Modules active | All five |
| Module errors | None |
| Landmark prediction | Drowsy |
| Landmark confidence | 0.831 |
| Landmark status | OK |
| Overall Driver Wellness Score | **49.8 / 100** |
| Risk Level | **Moderate Risk** |

This test demonstrates that all five modules can participate in the same pipeline and that their outputs can be successfully incorporated into the common risk score.

It is important to distinguish **pipeline validation** from **model accuracy validation**. The 49.8 score is an observed system output for the test clip and is not a ground-truth accuracy measure.

---

# 14. Failure Handling and Robustness

Milestone 6 improved failure handling across the integrated pipeline.

### 14.1 Landmark Failure

When the driver's face cannot be detected in the required window, the module now reports a meaningful unavailable/error state instead of silently disappearing.

Example:
`Face not detected in the required window`

### 14.2 Video-Length Failure

Insufficient video duration is handled gracefully rather than producing an invalid prediction.

### 14.3 Standardized Error States

Modules expose structured status information, allowing the dashboard and logs to distinguish:
- successful prediction;
- unavailable module;
- processing error.

### 14.4 Detection Robustness

The YOLO modules use filtering, NMS and/or temporal stabilization to reduce false positives and frame-to-frame flicker.

### 14.5 Streaming Robustness

The streaming orchestrator uses module-specific frame skipping and temporal histories to reduce unnecessary computation while maintaining recent predictions.

---

# 15. Gradio Application

The final user-facing application was implemented using **Gradio**.

## 15.1 Recorded Video Mode

The interface provides:
- video upload;
- Analyze button;
- processed/annotated video;
- module-wise predictions;
- risk values;
- overall wellness gauge;
- session summary.

## 15.2 Live Webcam Mode

The application also supports:
- webcam streaming;
- live annotations;
- real-time risk updates;
- session start/stop;
- session summary.

## 15.3 User Interface

The application uses a custom cockpit-style dashboard with:
- status indicators;
- risk gauge;
- visual module outputs;
- dark glassmorphism styling;
- dashboard-oriented information hierarchy.

---

# 16. Hugging Face Deployment

The project was prepared for deployment on **Hugging Face Spaces**.

Prepared artifacts include:
- `app.py`
- `requirements.txt`
- `README.md`
- `wellness_core.py`
- five model checkpoints
- module adapters
- Risk Fusion Engine
- Streaming Orchestrator

The architecture uses a thin application wrapper and a central core inference module.

## 16.1 Deployment Status

| Component | Status |
|---|---|
| `app.py` | Complete |
| `requirements.txt` | Complete |
| `README.md` | Complete |
| Local testing | Complete |
| Hugging Face Space | Created |
| Model upload | Complete |
| Runtime inference | Blocked by ZeroGPU quota |
| Deployment testing | Pending quota resolution |

The principal deployment challenges are:
- ZeroGPU quota;
- memory required to load all five models;
- 120-second GPU execution constraints for long videos.

Mitigation strategies include:
- model caching;
- lazy loading;
- frame skipping;
- lightweight architectures;
- future quantization/TensorRT optimization.

---

# 17. Cross-Module Performance Summary

| Module | Main Metric(s) | Final/Documented Result | Current Assessment |
|---|---|---|---|
| Video Fatigue | Test Accuracy / Macro F1 | 32.86% / 33.78% | Complementary signal; needs stronger validation |
| Landmark Fatigue | Integration result | Drowsy, confidence 0.831; system score 49.8 | Integrated; checkpoint discrepancy documented |
| Driver Activity | Test Accuracy / FPS | 93.14% / 80 FPS | Strong practical module |
| Smoking & Drinking | Precision / Recall / mAP@50 | 0.8465 / 0.8003 / 0.8197 | Strong real-time detector; drinking class weaker |
| Seatbelt & Phone | Precision / Recall / mAP@50 | 0.9370 / 0.9025 / 0.9526 | Strong documented validation performance |

**Important:** These metrics are not directly comparable because the modules perform different tasks and use different datasets, splits and evaluation protocols.

---

# 18. Major Technical Achievements in M6

### 18.1 Unified Modular Architecture

All five independently developed models were connected through standardized adapters and a common manager.

### 18.2 Standardized Prediction Contract

A common output structure was established so that the Risk Fusion Engine does not depend on model-specific internal representations.

### 18.3 Temporal Stabilization

Temporal processing was added where appropriate:
- BiLSTM sequence modeling for Video Fatigue;
- 45-frame windows and 10-window history for Landmark Fatigue;
- rolling probability smoothing for Driver Activity;
- temporal consensus/hysteresis for Seatbelt/Phone.

### 18.4 Risk Fusion

The system converts heterogeneous model outputs into a common driver-level score.

### 18.5 Robust Failure Handling

The final pipeline reports unavailable/error states instead of silently dropping failed modules.

### 18.6 User-Facing Application

A Gradio interface was developed for recorded-video and live-webcam operation.

### 18.7 Deployment Preparation

The project structure, dependencies, application wrapper and model artifacts were prepared for Hugging Face Spaces.

---

# 19. Project-Level Limitations

The following limitations remain important at the final milestone:

1. **Fatigue generalization:** Video Fatigue shows a large validation-to-test gap across unseen drivers.
2. **Landmark checkpoint discrepancy:** The deployed landmark checkpoint differs from the M5-evaluated checkpoint.
3. **Camera dependence:** Facial-landmark fatigue detection requires a sufficiently front-facing view.
4. **Live throughput:** Landmark processing can become the bottleneck in webcam mode.
5. **Drinking detection:** Drinking recall is substantially lower than smoking recall.
6. **Small-object localization:** Cigarettes and other small objects are challenging at strict IoU thresholds.
7. **Real-world domain shift:** Some datasets do not fully represent real vehicle cabins, lighting or mounting configurations.
8. **Deployment quota:** Hugging Face runtime inference is currently blocked by ZeroGPU quota.
9. **Multi-model resource use:** Loading five models simultaneously can require significant GPU memory.
10. **Safety-critical validation:** The integrated system has not undergone sufficient real-world validation to be treated as a safety-certified system.

---

# 20. Future Work

## 20.1 Model Improvements

- Increase cross-subject validation for fatigue models.
- Investigate stronger temporal architectures.
- Improve sequence/label alignment.
- Calibrate confidence scores.
- Validate landmark fusion on larger development splits.
- Improve drinking-class representation.
- Improve small-object detection through larger training resolution where compatible with the system contract.
- Evaluate quantization, pruning and distillation.

## 20.2 System Improvements

- Optimize MediaPipe landmark processing.
- Use asynchronous module execution where appropriate.
- Improve GPU memory management.
- Introduce lazy checkpoint loading.
- Add more efficient video decoding.
- Explore WebRTC for lower-latency live streaming.

## 20.3 Deployment Improvements

- Resolve Hugging Face ZeroGPU quota constraints.
- Evaluate a dedicated GPU-backed deployment.
- Add runtime health monitoring.
- Add model-version tracking.
- Add structured logging and inference telemetry.

## 20.4 Validation Improvements

- Test on diverse real-world cabin environments.
- Evaluate different camera mounting positions.
- Test night, glare, shadow and occlusion conditions.
- Perform demographic/per-group fairness audits where appropriate data is available.
- Establish independent end-to-end ground-truth evaluation.
- Conduct longer-duration live-video testing.

---

# 21. Final Deployment Readiness Assessment

| Area | Assessment |
|---|---|
| Five-model integration | **Complete** |
| Standardized module interfaces | **Complete** |
| Risk Fusion integration | **Complete** |
| Recorded-video pipeline | **Functional** |
| Live webcam pathway | **Functional with throughput limitations** |
| Gradio application | **Complete and locally functional** |
| Hugging Face Space | **Created and artifacts uploaded** |
| Hugging Face runtime inference | **Blocked by ZeroGPU quota** |
| Driver Activity readiness | **Strong** |
| Smoking/Drinking readiness | **Strong with documented limitations** |
| Seatbelt/Phone readiness | **Strong documented validation; further real-world validation recommended** |
| Landmark Fatigue readiness | **Integrated; checkpoint/camera limitations documented** |
| Video Fatigue readiness | **Experimental complementary signal** |
| Safety-critical deployment | **Not recommended without further validation** |

---

# 22. Final Conclusion

Milestone 6 completes the major engineering objective of the **AI-Powered Driver Wellness & Safety Monitoring System**: five independently developed driver-monitoring modules have been consolidated into a unified video-processing, risk-fusion and visualization pipeline.

The project demonstrates a modular architecture capable of combining:
- temporal video fatigue analysis;
- facial-landmark fatigue analysis;
- driver activity recognition;
- smoking/drinking detection;
- seatbelt/phone detection.

The final system supports standardized outputs, risk fusion, recorded-video analysis, live-webcam processing and a Gradio-based user interface. Integration testing confirmed that all five modules can operate together and contribute to a common Driver Wellness Score.

The strongest quantitative modules are the **Driver Activity Classification** and the two **YOLO-based behavior-detection modules**, while the fatigue modules remain more sensitive to dataset, camera and cross-subject generalization issues.

The final project should therefore be characterized as a **functionally integrated research/engineering prototype**, rather than a safety-certified production system. The architecture, model adapters, risk framework, user interface and deployment artifacts provide a strong foundation for future improvements in model generalization, runtime efficiency, real-world validation and deployment robustness.

---

# 23. Final Deliverables

The consolidated Milestone 6 submission consists of:

- Five trained model checkpoints.
- Five standardized model adapters.
- Common `wellness_core.py` inference architecture.
- Streaming Orchestrator.
- Driver Wellness Module Manager.
- Common Risk Fusion Engine.
- Gradio user interface.
- Recorded-video analysis pathway.
- Live-webcam pathway.
- Deployment configuration.
- `app.py`
- `requirements.txt`
- `README.md`
- Individual module documentation.
- Final consolidated Milestone 6 report.

---

# 24. Source Documents Used for Consolidation

This report was consolidated directly from the five submitted M6 module reports:

1. **Milestone6_Video_Fatigue_Final_Summary.md** — Video-Based Fatigue Detection — Kushagra Barodekar
2. **Milestone-6-Landmark-Fatigue-Integration.docx** — Landmark-Based Fatigue Detection — Shiwani Tiwari
3. **Milestone 6 Report(Shubham).docx** — Driver Activity Classification — Shubham
4. **Milestone6_Summary_Report (1).docx** — Smoking & Drinking Detection — Ravina
5. **Sohini_M6_Final_Individual_Report_Seatbelt_Phone_Usage_Detection (1).docx** — Seat Belt & Phone Usage Detection — Sohini Sarkar

**End of Milestone 6 Final Consolidated Report**
