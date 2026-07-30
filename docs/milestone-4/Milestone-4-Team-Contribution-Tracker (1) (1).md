# Milestone 4 Team Contribution Tracker

## AI-Powered Driver Wellness and Safety Monitoring System

This file tracks the work completed by each team member for **Milestone 4**.

> **Milestone 4 Focus:** Model training, evaluation, hyperparameter tuning, and optimization.
> Each member trains and evaluates the model designed in Milestone 3. All modules are made training-complete and results-ready so they are integration-ready for the final system in Milestone 5.

---

## Contribution Summary

| Team Member  | Responsibility | Contribution for Milestone 4 | Signature |
| ------------ | -------------- | ---------------------------- | --------- |
| **Kushagra** | Video-Based Fatigue Detection (Temporal Deep Learning) | - Implemented the CNN-LSTM training pipeline designed in Milestone 3.<br>- Prepared and preprocessed the video dataset (frame extraction, sequence generation, normalization).<br>- Trained the temporal model on the fatigue dataset (Safe / Caution / High Risk).<br>- Applied data augmentation and regularization (dropout) to reduce overfitting.<br>- Evaluated using Accuracy, Precision, Recall, F1-score, and confusion matrix.<br>- Saved trained model checkpoints and documented training/validation curves.<br>- Prepared the training notebook and report section. | KB |
| **Shiwani**  | Landmark-Based Temporal Analysis (EAR, MAR, Head Pose, Gaze) + Input/Output Specification Update & Final Report Integration | - Implemented the LSTM-based temporal training pipeline.<br>- Generated feature sequences (EAR, MAR, pitch, yaw, roll) using the sliding-window strategy.<br>- Trained the model on landmark sequences (Talking / Yawning / Normal).<br>- Handled class imbalance using class weighting / augmentation.<br>- Evaluated with Accuracy, Precision, Recall, and F1-score.<br>- Tuned hyperparameters and the temporal window size.<br>- Saved checkpoints and plotted training/validation curves.<br>- Updated the unified input/output specification and feature summary tables with actual trained results; prepared report section.<br>- Collected and merged all member report sections into a single, consistently formatted **Milestone-4-Report.md** with uniform numbering, inserted result plots, verified tables/references, and final proofreading. | ST |
| **Shubham**  | Driver Distraction / Activity Classification + Model Comparison | - Implemented the MobileNetV3 training pipeline.<br>- Preprocessed and augmented the driver activity dataset.<br>- Trained the classifier (Safe Driving, Talking on Phone, Texting on Phone, Turning, Other Activities).<br>- Evaluated with Accuracy, Precision, Recall, F1-score, and confusion matrix.<br>- Tuned hyperparameters and compared results against the baseline models.<br>- Recorded parameter count, FLOPs, and inference speed of the trained model.<br>- Collected training results/metrics from all modules into the combined results comparison table.<br>- Saved model checkpoints. | SB |
| **Sohini**   | Seat Belt Detection & Phone Usage Detection + Training Infrastructure | - Implemented the YOLOv8 training pipeline for both seat belt detection and phone usage detection.<br>- Prepared and verified/annotated the detection dataset (Seat Belt, Phone Usage).<br>- Trained the combined seat belt and phone usage detection model with augmentation.<br>- Evaluated with mAP@50, mAP@50–95, Precision, and Recall.<br>- Tuned hyperparameters (learning rate, batch size, epochs, augmentation settings).<br>- Set up the shared training infrastructure: checkpointing, early stopping, and logging for the team.<br>- Saved trained weights and exported detection results.<br>- Prepared report section. | SS |
| **Ravina**   | Smoking & Drinking Detection + Presentation, Contribution Tracker & Final Review | - Implemented the YOLOv8 training pipeline.<br>- Prepared and verified the detection dataset (Smoking, Drinking).<br>- Trained the detection model with augmentation.<br>- Evaluated with mAP@50, mAP@50–95, Precision, and Recall.<br>- Tuned hyperparameters and computational settings.<br>- Saved trained weights and documented results.<br>- Prepared and merged the Milestone-4 presentation, maintained the team contribution tracker, reviewed the final report, and prepared the submission checklist. | R |

---

## Common Team Responsibilities

| Team Member | Common Deliverable |
| ----------- | ------------------ |
| **Kushagra** | Video-based fatigue model training, training/validation curves and model checkpoints |
| **Shiwani**  | Updated input specification, output specification, and feature summary tables with trained results; final combined report integration — Milestone-4-Report.md, final report review, submission-ready markdown file |
| **Shubham**  | Overall results comparison table, computational metrics collection (parameters, FLOPs, inference speed) |
| **Sohini**   | Seat belt & phone usage detection training; shared training infrastructure (checkpointing, early stopping, logging), hyperparameter tuning setup |
| **Ravina**   | Milestone-4 presentation (.pdf), team contribution tracker (.md), submission checklist |

---

## End-to-End System Architecture

```
                        Camera Feed
                             │
                             ▼
 ──────────────────────────────────────────────────────
  Driver Activity Classification        (trained)
  Seat Belt Detection                   (trained)
  Smoking & Drinking Detection          (trained)
  Video-Based Fatigue Detection         (trained)
  Landmark-Based Fatigue Detection      (trained)
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
| Dataset preparation & preprocessing | ✔ | ✔ | ✔ | ✔ | ✔ |
| Data augmentation applied | ✔ | ✔ | ✔ | ✔ | ✔ |
| Model training completed | ✔ | ✔ | ✔ | ✔ | ✔ |
| Training / validation curves plotted | ✔ | ✔ | ✔ | ✔ | ✔ |
| Model evaluation (metrics) | ✔ | ✔ | ✔ | ✔ | ✔ |
| Confusion matrix / mAP reporting | ✔ | ✔ | ✔ | ✔ | ✔ |
| Hyperparameter tuning | ✔ | ✔ | ✔ | ✔ | ✔ |
| Overfitting handling / regularization | ✔ | ✔ | ✔ | ✔ | ✔ |
| Model checkpoint saved | ✔ | ✔ | ✔ | ✔ | ✔ |
| Trained model exported | ✔ | ✔ | ✔ | ✔ | ✔ |
| Results comparison table | ✔ | ✔ | ✔ | ✔ | ✔ |
| Report section | ✔ | ✔ | ✔ | ✔ | ✔ |
| Training notebook / script | ✔ | ✔ | ✔ | ✔ | ✔ |
| References | ✔ | ✔ | ✔ | ✔ | ✔ |
| Work log update | ✔ | ✔ | ✔ | ✔ | ✔ |
| Review initials / sign-off | KB | ST | SB | SS | R |

---

## Milestone 4 Submission Notes

The final submission will include:

1. **Milestone-4-Report.md** — prepared and integrated by Shiwani using report sections from all members
2. **Milestone-4-Presentation.pdf** — prepared and finalized by Ravina with contributions from all members
3. **Milestone-4-Team-Contribution-Tracker.md** — prepared and maintained by Ravina
4. Trained model checkpoints / weights for all five modules
5. Training and validation curves / evaluation plots
6. Results comparison table (metrics, parameters, FLOPs, inference speed)
7. Model training notebooks / scripts
8. Confusion matrices / mAP reports
9. References
10. Team review initials / sign-off

---

## Team Review & Sign-Off

| Team Member | Module Reviewed | Initials | Date |
| ----------- | --------------- | -------- | ---- |
| Kushagra | Video-Based Fatigue Detection | KB | |
| Shiwani | Landmark-Based Temporal Analysis + I/O Update + Combined Report | ST | |
| Shubham | Driver Activity Classification + Model Comparison | SB | |
| Sohini | Seat Belt & Phone Usage Detection + Training Infrastructure | SS | |
| Ravina | Smoking & Drinking Detection + Presentation & Tracker | R | |

*All team members confirm that the Milestone 4 model training and evaluation is complete and results-ready for final integration in Milestone 5.*
