# Milestone 5 Team Contribution Tracker

## AI-Powered Driver Wellness and Safety Monitoring System

This file tracks the work assigned to each team member for **Milestone 5**.

> **Milestone 5 Focus:** Evaluation of model performance and analysis of results.
> Each member evaluates the model they trained in Milestone 4 on a held-out evaluation set, reports quantitative and qualitative results, performs error analysis, and documents limitations. All modules are made evaluation-complete and analysis-ready so they are integration-ready for the final system in Milestone 5's successor / final report.

---

## Milestone 5 Task Reference

The submission must address the following, as applicable to each module:

1. Briefly restate the trained model and pipeline from Milestone 4.
2. Describe the evaluation dataset (size, composition, evaluation-time preprocessing).
3. Specify the evaluation environment (hardware, software frameworks, runtime setup) for reproducibility.
4. Define the performance metrics and justify why they are appropriate.
5. Present quantitative results (tables/plots), including comparisons across models, configurations, or hyperparameters.
6. Include visualizations (confusion matrices, ROC curves, PR curves, or task-specific plots).
7. Provide qualitative results (sample outputs: successes and failure cases).
8. Perform error analysis (patterns in mistakes and likely reasons).
9. Discuss key observations, limitations, and anomalies (gaps between expected and actual performance).

Every module owner completes tasks 1–9 for their own module. The distribution below additionally assigns the **cross-cutting / integration** responsibilities (a common evaluation protocol, combined comparison tables, unified plotting, the report, the presentation, and the tracker).

---

## Contribution Summary

| Team Member  | Responsibility | Contribution for Milestone 5 | Signature |
| ------------ | -------------- | ---------------------------- | --------- |
| **Kushagra** | Video-Based Fatigue Detection (Temporal Deep Learning) Evaluation + Common Evaluation Protocol | - Restated the trained CNN-LSTM model and the frame-extraction → sequence-generation → temporal-modeling pipeline from Milestone 4.<br>- Described the fatigue evaluation set (subject-disjoint held-out videos, Safe / Caution / High Risk composition, evaluation-time frame sampling and normalization).<br>- Evaluated with Accuracy, Precision, Recall, F1-score and a 3-class confusion matrix; added per-class ROC/PR curves.<br>- Reported quantitative results across the final model vs. the Milestone-3 candidate baselines (CNN-GRU / TCN).<br>- Provided qualitative clips: correct high-risk detections and failure cases (low light, occlusion, brief eye closure).<br>- Performed error analysis and documented limitations/anomalies.<br>- **Common:** defined the shared evaluation protocol (fixed test splits, metric definitions, decision thresholds, reporting template) used by all five modules for consistency. | KB |
| **Shiwani**  | Landmark-Based Temporal Analysis (EAR, MAR, Head Pose, Gaze) Evaluation | - Restated the trained LSTM temporal model and sliding-window feature pipeline (EAR, MAR, pitch, yaw, roll).<br>- Described the landmark evaluation set (Talking / Yawning / Normal composition, window generation and feature normalization at evaluation time).<br>- Evaluated with Accuracy, Precision, Recall, F1-score, confusion matrix, and per-class PR curves.<br>- Presented quantitative results with the effect of temporal window size / class-weighting on metrics.<br>- Provided qualitative sequences (successful and failed detections) and error analysis (e.g., talking vs. yawning confusion).<br>- Documented observations, limitations, and expected-vs-actual gaps. | ST |
| **Shubham**  | Driver Distraction / Activity Classification Evaluation + Overall Results Comparison | - Restated the trained MobileNetV3 classifier and preprocessing pipeline.<br>- Described the activity evaluation set (Safe Driving, Talking on Phone, Texting on Phone, Turning, Other Activities; driver-disjoint split; evaluation-time resize/normalize).<br>- Evaluated with Accuracy, Precision, Recall, F1-score, confusion matrix, and per-class ROC/PR curves.<br>- Presented quantitative results comparing the final model against ResNet50 / EfficientNet-B0 baselines (accuracy, params, FLOPs, inference speed).<br>- Provided qualitative samples (successes and failure cases) and error analysis (e.g., texting vs. talking confusion).<br>- Discussed limitations and anomalies.<br>- **Common:** compiled the **overall results comparison table** across all five modules (metrics + parameters + FLOPs + inference speed) and the consolidated computational-cost summary. | SB |
| **Sohini**   | Seat Belt & Phone Usage Detection Evaluation + Common Evaluation Environment & Plotting | - Restated the trained YOLOv8 detection model and pipeline for seat belt and phone usage.<br>- Described the detection evaluation set (Seat Belt, Phone classes; image counts; evaluation-time letterbox/resize and `data.yaml` test config).<br>- Evaluated with mAP@50, mAP@50–95, Precision, Recall, and PR curves per class.<br>- Presented quantitative results across YOLOv8n / YOLO11n / YOLOv8s configurations.<br>- Provided qualitative detections (true positives, missed/false detections) and error analysis (small objects, occlusion, seat-belt-color confusion).<br>- Documented limitations and anomalies.<br>- **Common:** documented the shared **evaluation environment** (GPU/CPU, CUDA, PyTorch/Ultralytics versions, runtime setup) for reproducibility and prepared the unified plotting utilities (confusion matrices, ROC/PR curves) used by all modules. | SS |
| **Ravina**   | Smoking & Drinking Detection Evaluation + Presentation, Contribution Tracker & Final Review | - Restated the trained YOLOv8 detection model and pipeline for smoking and drinking.<br>- Described the detection evaluation set (Smoking, Drinking classes; image counts; evaluation-time preprocessing).<br>- Evaluated with mAP@50, mAP@50–95, Precision, Recall, and PR curves.<br>- Presented quantitative results across candidate configurations and provided qualitative detections (successes and failures).<br>- Performed error analysis (hand-to-mouth ambiguity, phone-vs-cigarette confusion) and documented limitations/anomalies.<br>- **Common:** collected and merged all member evaluation sections into a single, consistently formatted **Milestone-5-Report.md** with uniform numbering, inserted result plots, verified tables/references, and final proofreading; prepared and merged the Milestone-5 presentation, maintained the team contribution tracker, reviewed the final report, and prepared the submission checklist. | R |

---

## Common Team Responsibilities

| Team Member | Common Deliverable |
| ----------- | ------------------ |
| **Kushagra** | Shared evaluation protocol (fixed test splits, metric definitions, thresholds, reporting template) |
| **Shiwani**  | Landmark-based fatigue detection module evaluation, confusion matrix and PR-curve plots |
| **Shubham**  | Overall results comparison table + consolidated computational-cost summary (metrics, parameters, FLOPs, inference speed) |
| **Sohini**   | Shared evaluation environment documentation (hardware/software/runtime) + unified plotting utilities (confusion matrices, ROC/PR curves) |
| **Ravina**   | Final combined report integration — Milestone-5-Report.md, final report review, submission-ready markdown file; Milestone-5 presentation (.pdf), team contribution tracker (.md), submission checklist |

---

## Milestone 4 Review Feedback — Action Items Carried Into Milestone 5

The following issues were raised in the Milestone 4 review meeting. Each item is assigned an owner and must be resolved and documented as part of Milestone 5.

### A. Architecture & Integration Strategy

| # | Action Item | Owner(s) | Status |
| - | ----------- | -------- | ------ |
| A1 | Convert the integrated pipeline from a static, batch video-processing loop into a **true real-time streaming pipeline**. | All (integration: Kushagra + Ravina) | ☐ |
| A2 | Implement a **live sliding-window simulation** where temporal frame buffers dynamically drop the oldest frames and add incoming frames to continuously update the driver score in real time. | Kushagra, Shiwani | ☐ |
| A3 | Add a **contextual logic gate for stationary drivers** — a state mechanism (simulated vehicle movement / ignition status) that bypasses wellness-score calculation when the vehicle is stationary (e.g., a driver resting in a parked car should not be penalized with high risk). | All (integration: Kushagra + Ravina) | ☐ |

### B. Conceptual & Mathematical Corrections

| # | Action Item | Owner(s) | Status |
| - | ----------- | -------- | ------ |
| B1 | **Landmark module:** Correct the erroneous claim that the model was trained *without a loss function*. Identify, use, and document the exact loss function in the code pipeline (e.g., Cross-Entropy Loss). | Shiwani | ☐ |
| B2 | **Activity module:** Correct the definition of **Softmax** — it maps a vector of raw logits to a probability distribution summing to one (not a way to "remove losses from the pipeline"). | Shubham | ☐ |
| B3 | **Activity module:** Mathematically correct the explanations of **weight decay** (L2 regularization penalizing large weights to reduce overfitting) and the **backpropagation step** of the optimizer; clarify the distinction between input- and hidden-layer weights. | Shubham | ☐ |
| B4 | **Landmark module:** Write out the **EAR and MAR formulas in standard LaTeX** (not black-box generated code) and demonstrate full understanding. | Shiwani | ☐ |
| B5 | **Landmark module:** Provide **statistical / literature-based justification** for the 15% yawning threshold used to trigger the "Drowsy" state, and for the rule-based alert-mapping thresholds. | Shiwani | ☐ |

### C. Training & Data Issues

| # | Action Item | Owner(s) | Status |
| - | ----------- | -------- | ------ |
| C1 | **Video-Based Fatigue module:** The 1-epoch run gave 31.78% validation accuracy — below the 33.33% random baseline for 3-class classification. Complete **full training to convergence** using **persistent checkpoint-saving** (track/reload model weight dictionaries across runs) to overcome local/session runtime limits. | Kushagra | ☐ |
| C2 | **Activity module:** Analyze the **data-balancing paradox** — full balancing dropped validation accuracy from 80% to 24%. Determine whether the original high performance was a **data leak across splits** or **severe minority-class underfitting**. | Shubham | ☐ |
| C3 | **Activity module:** Eliminate class redundancy with the YOLO-based phone-detection module. | Shubham | ☐ |
| C4 | **Smoking & Drinking module:** Optimize detection thresholds to **minimize high false-positive rates** (e.g., normal hand movement classified as "drinking"). | Ravina | ☐ |
| C5 | Explicitly document a **distinct, unseen test split** (separate from validation) for the Milestone 5 evaluations. | All (protocol: Kushagra) | ☐ |

### D. Documentation & Visualization Standards

| # | Action Item | Owner(s) | Status |
| - | ----------- | -------- | ------ |
| D1 | Add the missing visualizations to the report/slides: **training & validation loss curves, learning curves (epochs vs. accuracy/mAP), and final validation confusion matrices**. | All owners; consolidation: Sohini | ☐ |
| D2 | Add a dedicated section of **qualitative run-time inference examples** showing successful predictions and failure cases (e.g., the false "drinking" detection from the demo). | All owners; consolidation: Ravina | ☐ |
| D3 | Resolve **metric discrepancies between slides and report** (e.g., mismatched 94% vs. 90% accuracy figures). | Shubham; final check: Ravina | ☐ |
| D4 | Include **formal initials / review signatures** confirming peer review of each module before submission. | All; tracker: Ravina | ☐ |

### E. Individual Feedback & Follow-Ups

| Member | Strength Noted | Required Follow-Up for Milestone 5 | Status |
| ------ | -------------- | ---------------------------------- | ------ |
| **Kushagra** | Structured the sequential preprocessing pipeline and uniform frame sampling well. | Address sub-random baseline performance; implement robust checkpoint-saving to complete multiple epochs; establish a clean, distinct test partition separate from validation. | ☐ |
| **Shiwani** | Successfully balanced the dataset by removing the underrepresented class. | Resolve the loss-function misunderstanding; document EAR & MAR formulas mathematically (LaTeX); provide quantitative justification for rule-based alert thresholds. | ☐ |
| **Shubham** | Explained MobileNetV3 efficiency–accuracy tradeoffs for edge hardware effectively. | Correctly define Softmax, weight decay, and input-vs-hidden-layer weights; resolve the 24% balancing accuracy drop; eliminate class redundancy with YOLO phone detection; align accuracy values between slides and report. | ☐ |
| **Sohini** | Explained detection metrics (mAP@0.5, mAP@0.5:0.95) and optimization schedules clearly. | Describe the exact layer configurations and frozen-weight boundaries of the transfer-learning pipeline. | ☐ |
| **Ravina** | — | **Absent from the mandatory review meeting → score of zero for Milestone 4.** Must optimize model thresholds to minimize false positives (e.g., normal hand movement as "drinking"); ensure attendance and active engagement in all future reviews. | ☐ |

---

## End-to-End System Architecture

```
                        Camera Feed
                             │
                             ▼
 ──────────────────────────────────────────────────────
  Driver Activity Classification        (trained + evaluated)
  Seat Belt & Phone Usage Detection     (trained + evaluated)
  Smoking & Drinking Detection          (trained + evaluated)
  Video-Based Fatigue Detection         (trained + evaluated)
  Landmark-Based Fatigue Detection      (trained + evaluated)
 ──────────────────────────────────────────────────────
                             │
                             ▼
                     Risk Fusion Engine
                             │
                             ▼
                   Driver Wellness Score
                             │
                             ▼
                   Driver Safety Report
                             │
                             ▼
              Uber / Ola / Rapido Dashboard
```

---

## Per-Member Completion Checklist

Each feature owner is responsible for completing the following items for their module:

| Item | Kushagra | Shiwani | Shubham | Sohini | Ravina |
| ---- | -------- | ------- | ------- | ------ | ------ |
| Restate trained model & pipeline | ✔ | ✔ | ✔ | ✔ | ✔ |
| Evaluation dataset described (size/composition/preprocessing) | ✔ | ✔ | ✔ | ✔ | ✔ |
| Evaluation environment specified (hardware/software/runtime) | ✔ | ✔ | ✔ | ✔ | ✔ |
| Metrics defined & justified | ✔ | ✔ | ✔ | ✔ | ✔ |
| Quantitative results (tables/plots) | ✔ | ✔ | ✔ | ✔ | ✔ |
| Model / configuration / hyperparameter comparison | ✔ | ✔ | ✔ | ✔ | ✔ |
| Confusion matrix / mAP reporting | ✔ | ✔ | ✔ | ✔ | ✔ |
| ROC / PR curves (or task-specific plots) | ✔ | ✔ | ✔ | ✔ | ✔ |
| Qualitative results (success + failure cases) | ✔ | ✔ | ✔ | ✔ | ✔ |
| Error analysis (patterns + reasons) | ✔ | ✔ | ✔ | ✔ | ✔ |
| Observations, limitations & anomalies | ✔ | ✔ | ✔ | ✔ | ✔ |
| Report section | ✔ | ✔ | ✔ | ✔ | ✔ |
| Evaluation notebook / script | ✔ | ✔ | ✔ | ✔ | ✔ |
| References | ✔ | ✔ | ✔ | ✔ | ✔ |
| Work log update | ✔ | ✔ | ✔ | ✔ | ✔ |
| Review initials / sign-off | KB | ST | SB | SS | R |

---

## Milestone 5 Submission Notes

The final submission will include:

1. **Milestone-5-Report.md** — prepared and integrated by Ravina using evaluation sections from all members
2. **Milestone-5-Presentation.pdf** — prepared and finalized by Ravina with contributions from all members
3. **Milestone-5-Team-Contribution-Tracker.md** — prepared and maintained by Ravina
4. Evaluation notebooks / scripts for all five modules
5. Quantitative results and overall comparison table (metrics, parameters, FLOPs, inference speed)
6. Evaluation plots (confusion matrices, ROC curves, PR curves, mAP reports)
7. Qualitative result samples (success and failure cases)
8. Error analysis and limitations write-up
9. References
10. Team review initials / sign-off

---

## Team Review & Sign-Off

| Team Member | Module Reviewed | Initials | Date |
| ----------- | --------------- | -------- | ---- |
| Kushagra | Video-Based Fatigue Detection Evaluation + Evaluation Protocol | KB | |
| Shiwani | Landmark-Based Temporal Analysis Evaluation | ST | |
| Shubham | Driver Activity Classification Evaluation + Overall Comparison | SB | |
| Sohini | Seat Belt & Phone Usage Detection Evaluation + Evaluation Environment & Plotting | SS | |
| Ravina | Smoking & Drinking Detection Evaluation + Combined Report + Presentation & Tracker | R | |

*All team members confirm that the Milestone 5 model evaluation and results analysis is complete and integration-ready for the final system.*
