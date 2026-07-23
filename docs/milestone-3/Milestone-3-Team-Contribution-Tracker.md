# Milestone 3 Team Contribution Tracker

## AI-Powered Driver Wellness and Safety Monitoring System

This file tracks the work completed by each team member for **Milestone 3**.

> **Milestone 3 Focus:** Model architecture design, model selection, and system planning.
> No model training is performed in this milestone. All modules must be designed and documented so that they are implementation-ready for training in Milestone 4.

---

## Contribution Summary

| Team Member  | Responsibility | Contribution for Milestone 3 | Signature |
| ------------ | -------------- | ---------------------------- | --------- |
| **Kushagra** | Video-Based Fatigue Detection (Temporal Deep Learning) | - Compared candidate architectures (CNN-LSTM, CNN-GRU, TCN, Lightweight 3D CNN) and selected the final model with justification.<br>- Designed frame extraction, sequence generation, and temporal modeling pipelines.<br>- Defined input specification (16 × 224 × 224 × 3) and output classes (Safe / Caution / High Risk).<br>- Selected loss function, evaluation metrics, hyperparameter plan, and computational requirement estimates.<br>- Prepared architecture diagram, pipeline diagram, input/output table, notebook skeleton, references, and report section. | KB |
| **Shiwani**  | Landmark-Based Temporal Analysis (EAR, MAR, Head Pose, Gaze) + Input/Output Specification Summary | - Performed feature engineering for EAR, MAR, pitch, yaw, and roll.<br>- Compared LSTM, GRU, TCN, and MLP baseline architectures and selected the final model.<br>- Designed temporal sequence structure and sliding window strategy with input specification (30, 5) and outputs (Talking / Yawning / Normal).<br>- Selected CrossEntropyLoss and evaluation metrics (Accuracy, Precision, Recall, F1-score); prepared hyperparameter plan and computational estimates.<br>- Prepared architecture diagram, sequence pipeline, input/output table, model justification, notebook skeleton, references, and report section.<br>- Compiled unified input/output specification tables and the feature summary table across all modules. | ST |
| **Shubham**  | Driver Distraction / Activity Classification + Model Comparison, Computational Analysis & Final Report Integration | - Compared MobileNetV3, ResNet50, and EfficientNet-B0 on parameter count, FLOPs, inference speed, and expected accuracy.<br>- Selected final model with architecture justification; defined input (3, 224, 224) and output classes (Safe Driving, Talking on Phone, Texting on Phone, Turning, Other Activities).<br>- Selected CrossEntropyLoss and metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix).<br>- Planned hyperparameters (learning rate, batch size, epochs, optimizer, scheduler) and computational requirements.<br>- Prepared architecture diagram, model comparison table, input/output specification, notebook skeleton, references, and report section.<br>- Collected parameter counts, FLOPs, memory requirements, and inference-speed estimates from all modules into the overall comparison table and computational requirements section.<br>- Collected and merged all member report sections into a single, consistently formatted **Milestone-3-Report.md** with uniform numbering, inserted diagrams, verified tables/references, and final proofreading. | SB |
| **Sohini**   | Seat Belt & Phone Usage Detection + Training Strategy for Milestone 4 | - Compared YOLOv8n, YOLO11n, and YOLOv8s on parameters, mAP, inference speed, and GPU requirements; selected the final detection model.<br>- Defined input (640 × 640 RGB) and output classes (Phone, Seat Belt).<br>- Selected losses (Box Loss, Classification Loss, Distribution Focal Loss) and metrics (mAP@50, mAP@50–95, Precision, Recall).<br>- Prepared architecture diagram, YOLO pipeline, hyperparameter plan, model justification, notebook skeleton, references, and report section.<br>- Defined the Milestone-4 training strategy: training pipelines, optimizer selection, LR scheduler, augmentation, checkpointing, and early stopping, with a hyperparameter summary table. | SS |
| **Ravina**   | Smoking & Drinking Detection + Presentation, Contribution Tracker & Final Review | - Compared YOLOv8n, YOLO11n, and YOLOv8s and selected the final model with justification.<br>- Defined input (640 × 640 RGB) and output classes (Smoking, Drinking).<br>- Selected YOLO detection loss and metrics (mAP@50, mAP@50–95, Precision, Recall); prepared hyperparameter plan and computational requirements.<br>- Prepared architecture diagram, model justification, notebook skeleton, references, and report section.<br>- Prepared and merged the Milestone-3 presentation, maintained the team contribution tracker, reviewed the final report, and prepared the submission checklist. | R |

---

## Common Team Responsibilities

| Team Member | Common Deliverable |
| ----------- | ------------------ |
| **Kushagra** | Video-based fatigue detection module design, architecture and pipeline diagrams |
| **Shiwani**  | Input specification table, output specification table, feature summary table |
| **Shubham**  | Overall model comparison table, computational requirements section, final combined report integration — Milestone-3-Report.md, final report review, submission-ready markdown file |
| **Sohini**   | Milestone-4 training plan, hyperparameter summary table |
| **Ravina**   | Milestone-3 presentation (.pdf), team contribution tracker (.md), submission checklist |

---

## End-to-End System Architecture

```
                        Camera Feed
                             │
                             ▼
 ──────────────────────────────────────────────────────
  Driver Activity Classification
  Seat Belt Detection
  Smoking & Drinking Detection
  Video-Based Fatigue Detection
  Landmark-Based Fatigue Detection
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
| Final model selection | ✔ | ✔ | ✔ | ✔ | ✔ |
| Baseline model comparison | ✔ | ✔ | ✔ | ✔ | ✔ |
| Architecture justification | ✔ | ✔ | ✔ | ✔ | ✔ |
| Candidate model comparison | ✔ | ✔ | ✔ | ✔ | ✔ |
| Input specification | ✔ | ✔ | ✔ | ✔ | ✔ |
| Output specification | ✔ | ✔ | ✔ | ✔ | ✔ |
| Data flow explanation | ✔ | ✔ | ✔ | ✔ | ✔ |
| Loss function selection | ✔ | ✔ | ✔ | ✔ | ✔ |
| Evaluation metric selection | ✔ | ✔ | ✔ | ✔ | ✔ |
| Hyperparameter planning | ✔ | ✔ | ✔ | ✔ | ✔ |
| Computational requirement estimation | ✔ | ✔ | ✔ | ✔ | ✔ |
| End-to-end inference pipeline | ✔ | ✔ | ✔ | ✔ | ✔ |
| Architecture diagram | ✔ | ✔ | ✔ | ✔ | ✔ |
| Report section | ✔ | ✔ | ✔ | ✔ | ✔ |
| Notebook / script skeleton | ✔ | ✔ | ✔ | ✔ | ✔ |
| References | ✔ | ✔ | ✔ | ✔ | ✔ |
| Work log update | ✔ | ✔ | ✔ | ✔ | ✔ |
| Review initials / sign-off | KB | ST | SB | SS | R |

---

## Milestone 3 Submission Notes

The final submission will include:

1. **Milestone-3-Report.md** — prepared and integrated by Shubham using report sections from all members
2. **Milestone-3-Presentation.pdf** — prepared and finalized by Ravina with contributions from all members
3. **Milestone-3-Team-Contribution-Tracker.md** — prepared and maintained by Ravina
4. Architecture diagrams for all five modules
5. End-to-End System Architecture diagram
6. Model comparison tables
7. Model design notebooks / script skeletons
8. Training strategy for Milestone 4
9. References
10. Team review initials / sign-off

---

## Team Review & Sign-Off

| Team Member | Module Reviewed | Initials | Date |
| ----------- | --------------- | -------- | ---- |
| Kushagra | Video-Based Fatigue Detection | KB | |
| Shiwani | Landmark-Based Temporal Analysis + I/O Summary | ST | |
| Shubham | Driver Activity Classification + Model Comparison + Combined Report | SB | |
| Sohini | Seat Belt & Phone Detection + Training Strategy | SS | |
| Ravina | Smoking & Drinking Detection + Presentation & Tracker | R | |

*All team members confirm that the Milestone 3 architecture design is complete and implementation-ready for training and integration in Milestone 4.*
