# AI-Powered Driver Wellness & Safety Monitoring System
## Final Team Contribution Summary — Milestones 1–6

**Purpose:** Final project submission  
**Coverage:** Milestones M1, M2, M3, M4, M5 and M6  
**Team Members:** Kushagra, Shiwani, Shubham, Sohini, Ravina

---

## 1. Purpose and Basis of This Document

This document provides a consolidated record of the contributions of each team member across **all six project milestones**.

It has been prepared from the team's milestone contribution trackers for M1–M6. The trackers distinguish between individual technical ownership and shared responsibilities such as report preparation, evaluation, integration, presentation, testing and deployment.

The Milestone 6 tracker explicitly requires each individual consolidated report to contain:

- M1 contribution
- M2 contribution
- M3 contribution
- M4 contribution
- M5 contribution
- M6 contribution
- Overall technical contribution
- Integration contribution
- Deployment contribution
- Artifacts/code produced
- Limitations
- Future improvements

This document follows that structure.

> **Transparency note:** Where a milestone tracker records only an assigned responsibility and does not contain a detailed description of completed work, this document does **not invent additional work**. Such cases are explicitly marked as "responsibility recorded; detailed contribution not documented in the tracker."

---

# 2. Milestone-Wise Contribution Overview

| Member | M1 | M2 | M3 | M4 | M5 | M6 |
|---|---|---|---|---|---|---|
| **Kushagra** | Problem definition, scope, stakeholders, documentation, slides, work distribution | Video fatigue dataset, UTA-RLDD, EDA, preprocessing, split strategy, GitHub structure | Video fatigue architecture and temporal model design | CNN-LSTM training and evaluation | Video fatigue evaluation + common evaluation protocol | Video fatigue integration, inference validation, integration testing |
| **Shiwani** | Dataset research and understanding | Landmark fatigue / processed dataset hosting summary *(detailed work not documented in tracker)* | Landmark temporal feature engineering, architecture, I/O specifications | LSTM landmark training, class balancing, tuning, report integration | Landmark fatigue evaluation | Landmark integration, feature validation, temporal verification, deployment support |
| **Shubham** | Deep-learning and YOLO literature review | Driver activity dataset / preprocessing / model-readiness summary *(detailed work not documented in tracker)* | Driver activity architecture, model comparison, computational analysis, final report integration | MobileNetV3 training, evaluation, model comparison | Driver activity evaluation + overall results comparison | Driver Activity integration and module verification |
| **Sohini** | Rule-based, MediaPipe/OpenCV literature review | Seat belt + phone dataset and data-quality/leakage-prevention summary *(detailed work not documented in tracker)* | Seat belt/phone model selection and M4 training strategy | YOLOv8 seat belt/phone training + shared training infrastructure | Seat belt/phone evaluation + common evaluation environment and plotting | Seat belt/phone integration, validation, deployment coordination, contribution tracking and final documentation |
| **Ravina** | Evaluation plan and metrics | Smoking/drinking dataset, presentation, work log and final review responsibilities *(detailed work not documented in tracker)* | Smoking/drinking architecture + presentation/tracker | YOLOv8 smoking/drinking training + presentation/tracker | Smoking/drinking evaluation + final report/presentation/tracker | Smoking/drinking integration + initial Lightning ai deployment documentation/support |

---

# 3. Milestone 1 — Problem Definition, Research and Evaluation Planning

## Kushagra
**Responsibility:** Problem Definition, Motivation, Scope, Stakeholders, Documentation, Slides and Work Distribution.

**Documented contribution:**
- Prepared the introduction.
- Defined the problem and motivation.
- Defined project scope.
- Conducted stakeholder analysis.
- Coordinated the documentation structure.
- Supported slide preparation and work distribution.

## Shiwani
**Responsibility:** Dataset Research and Dataset Understanding.

**Documented contribution:**
- Studied the Roboflow dataset.
- Examined dataset organization.
- Studied class labels and annotation format.
- Reviewed dataset statistics.
- Identified dataset strengths and limitations.

## Sohini
> The M1 tracker records the name as **"Sohin"**; later milestone trackers use **"Sohini"**. This consolidated document uses the later consistent name, Sohini.

**Responsibility:** Literature Review — Rule-Based and MediaPipe/OpenCV Methods.

**Documented contribution:**
- Researched EAR.
- Researched MAR.
- Researched blink rate.
- Researched PERCLOS.
- Researched head-pose estimation.
- Studied MediaPipe and OpenCV.
- Studied rule-based decision logic.

## Shubham
**Responsibility:** Literature Review — Deep Learning and YOLO-Based Methods.

**Documented contribution:**
- Researched YOLO-based object detection.
- Studied real-time detection.
- Reviewed lightweight YOLO variants.
- Studied challenges associated with YOLO.
- Studied YOLO–MediaPipe integration.

## Ravina
**Responsibility:** Evaluation Plan and Metrics.

**Documented contribution:**
- Prepared the evaluation plan.
- Defined metrics including precision, recall, F1-score, mAP and FPS.
- Considered false-alarm rate.
- Defined expected outputs.
- Defined testing scenarios.

---

# 4. Milestone 2 — Dataset Preparation, EDA and Preprocessing

Milestone 2 focused on dataset documentation, EDA, preprocessing, processed dataset structure, split strategy and report-ready dataset sections.

## Kushagra
**Responsibility:** Video-Based Fatigue Detection Dataset, UTA-RLDD Data Preparation, EDA, Preprocessing, Split Strategy and GitHub Structure.

**Documented contribution:**
- Selected and documented UTA-RLDD for fatigue detection.
- Verified 12 subjects and 36 videos.
- Generated EDA charts.
- Prepared sample grids and quality notes.
- Prepared subject-level splitting.
- Created the processed dataset structure.
- Extracted 180 sample frames.
- Prepared report-ready content.

## Shiwani
**Recorded responsibility:** Landmark-Based Temporal Analysis and Processed Dataset Hosting Summary.

**Documentation status:** The M2 contribution tracker records the responsibility, but does not provide a detailed description of the completed work for this member.

Therefore, no additional M2 activity is asserted here beyond the documented responsibility.

## Shubham
**Recorded responsibility:** Driver Distraction / Activity Classification and Preprocessing / Model-Readiness Summary.

**Documentation status:** The M2 contribution tracker records the responsibility, but does not provide a detailed description of the completed work.

Therefore, no additional M2 activity is asserted here beyond the documented responsibility.

## Sohini
**Recorded responsibility:** Seat Belt + Phone Usage Detection and Data Quality / Leakage Prevention Summary.

**Documentation status:** The M2 contribution tracker records the responsibility, but does not provide a detailed description of the completed work.

Therefore, no additional M2 activity is asserted here beyond the documented responsibility.

## Ravina
**Recorded responsibility:** Smoking and Drinking Detection, Presentation, Work Log, Final Review and Submission Checklist.

**Documentation status:** The M2 contribution tracker records these responsibilities, but does not provide a detailed description of the completed work.

Therefore, no additional M2 activity is asserted here beyond the documented responsibility.

---

# 5. Milestone 3 — Model Architecture, Model Selection and System Planning

Milestone 3 focused on selecting architectures, defining model inputs/outputs, designing inference pipelines and making all modules implementation-ready for training.

## Kushagra — Video-Based Fatigue Detection

**Documented contribution:**
- Compared CNN-LSTM, CNN-GRU, TCN and lightweight 3D CNN architectures.
- Selected the final architecture with justification.
- Designed frame extraction and sequence-generation pipelines.
- Designed temporal modelling.
- Defined input shape: `16 × 224 × 224 × 3`.
- Defined output classes: Safe / Caution / High Risk.
- Selected loss function and evaluation metrics.
- Planned hyperparameters.
- Estimated computational requirements.
- Prepared architecture and pipeline diagrams.
- Prepared input/output tables, notebook skeleton, references and report section.

## Shiwani — Landmark-Based Temporal Analysis

**Documented contribution:**
- Performed feature engineering for EAR, MAR, pitch, yaw and roll.
- Compared LSTM, GRU, TCN and MLP baselines.
- Selected the final architecture.
- Designed temporal sequences and sliding-window strategy.
- Defined input specification `(30, 5)`.
- Defined outputs: Talking / Yawning / Normal.
- Selected CrossEntropyLoss.
- Selected evaluation metrics.
- Prepared hyperparameter and computational plans.
- Prepared architecture/sequence diagrams, I/O table, model justification, notebook skeleton, references and report section.
- Compiled unified input/output specifications and the feature summary table across modules.

## Shubham — Driver Activity Classification

**Documented contribution:**
- Compared MobileNetV3, ResNet50 and EfficientNet-B0.
- Compared parameter count, FLOPs, inference speed and expected accuracy.
- Selected the final architecture.
- Defined input `(3, 224, 224)`.
- Defined output classes: Safe Driving, Talking on Phone, Texting on Phone, Turning and Other Activities.
- Selected CrossEntropyLoss and evaluation metrics.
- Planned hyperparameters and computational requirements.
- Prepared architecture diagram, model comparison table, I/O specification, notebook skeleton, references and report section.
- Collected parameter counts, FLOPs, memory requirements and inference-speed estimates across modules.
- Integrated all member report sections into the final `Milestone-3-Report.md`.
- Performed formatting, diagram insertion, table/reference verification and proofreading.

## Sohini — Seat Belt & Phone Usage Detection

**Documented contribution:**
- Compared YOLOv8n, YOLO11n and YOLOv8s.
- Compared parameters, mAP, inference speed and GPU requirements.
- Selected the final detection model.
- Defined `640 × 640 RGB` input.
- Defined Phone and Seat Belt output classes.
- Selected Box Loss, Classification Loss and Distribution Focal Loss.
- Selected mAP@50, mAP@50–95, Precision and Recall.
- Prepared architecture diagram and YOLO pipeline.
- Prepared hyperparameter plan and model justification.
- Prepared notebook skeleton, references and report section.
- Defined the Milestone 4 training strategy including optimizer, learning-rate scheduler, augmentation, checkpointing and early stopping.

## Ravina — Smoking & Drinking Detection + Presentation

**Documented contribution:**
- Compared YOLOv8n, YOLO11n and YOLOv8s.
- Selected the final model with justification.
- Defined `640 × 640 RGB` input.
- Defined Smoking and Drinking output classes.
- Selected YOLO loss and evaluation metrics.
- Prepared hyperparameter and computational plans.
- Prepared architecture diagram, model justification, notebook skeleton, references and report section.
- Prepared and merged the Milestone 3 presentation.
- Maintained the contribution tracker.
- Reviewed the final report.
- Prepared the submission checklist.

---

# 6. Milestone 4 — Model Training, Evaluation, Tuning and Optimization

Milestone 4 focused on implementing and training the architectures selected in M3.

## Kushagra — Video Fatigue
- Implemented the CNN-LSTM training pipeline.
- Prepared video preprocessing, frame extraction, sequence generation and normalization.
- Trained the Safe / Caution / High Risk temporal model.
- Applied augmentation and dropout regularization.
- Evaluated using Accuracy, Precision, Recall, F1-score and confusion matrix.
- Saved checkpoints.
- Documented training/validation curves.
- Prepared training notebook and report section.

## Shiwani — Landmark Fatigue
- Implemented the LSTM temporal training pipeline.
- Generated EAR, MAR, pitch, yaw and roll sequences using sliding windows.
- Trained Talking / Yawning / Normal classification.
- Handled class imbalance using class weighting/augmentation.
- Evaluated Accuracy, Precision, Recall and F1-score.
- Tuned hyperparameters and temporal window size.
- Saved checkpoints.
- Plotted training/validation curves.
- Updated unified I/O and feature-summary tables with trained results.
- Integrated all member sections into `Milestone-4-Report.md`.
- Performed formatting, plot insertion, table/reference verification and proofreading.

## Shubham — Driver Activity
- Implemented the MobileNetV3 training pipeline.
- Preprocessed and augmented the driver activity dataset.
- Trained the activity classifier.
- Evaluated Accuracy, Precision, Recall, F1-score and confusion matrix.
- Tuned hyperparameters.
- Compared results against baselines.
- Recorded parameters, FLOPs and inference speed.
- Collected training results from all modules into the comparison table.
- Saved model checkpoints.

## Sohini — Seat Belt & Phone
- Implemented YOLOv8 training for seat belt and phone usage.
- Prepared and verified/annotated the detection dataset.
- Trained the combined detection model with augmentation.
- Evaluated mAP@50, mAP@50–95, Precision and Recall.
- Tuned learning rate, batch size, epochs and augmentation settings.
- Established shared training infrastructure for checkpointing, early stopping and logging.
- Saved trained weights and exported detection results.
- Prepared the report section.

## Ravina — Smoking & Drinking
- Implemented the YOLOv8 training pipeline.
- Prepared and verified the Smoking/Drinking dataset.
- Trained with augmentation.
- Evaluated mAP@50, mAP@50–95, Precision and Recall.
- Tuned hyperparameters and computational settings.
- Saved trained weights.
- Documented results.
- Prepared and merged the M4 presentation.
- Maintained the contribution tracker.
- Reviewed the final report.
- Prepared the submission checklist.

### Common M4 responsibilities
All five members were recorded as completing the module-level activities including dataset preparation, augmentation, training, curves, evaluation, confusion matrix/mAP reporting, tuning, regularization, checkpointing, export, comparison tables, report sections, training scripts, references and work logs.

---

# 7. Milestone 5 — Model Evaluation, Error Analysis and Results

Milestone 5 focused on held-out evaluation, quantitative/qualitative results, error analysis, limitations and deployment readiness.

## Kushagra
**Video Fatigue Evaluation + Common Evaluation Protocol**
- Evaluated the trained CNN-LSTM pipeline.
- Used subject-disjoint held-out videos.
- Evaluated Accuracy, Precision, Recall and F1-score.
- Produced a 3-class confusion matrix and per-class ROC/PR curves.
- Compared the final model with CNN-GRU/TCN candidates.
- Provided qualitative successful/failure clips.
- Analysed failure cases such as low light, occlusion and brief eye closure.
- Defined the shared evaluation protocol used across modules: fixed test splits, metric definitions, thresholds and reporting template.

## Shiwani
**Landmark Fatigue Evaluation**
- Evaluated the trained LSTM landmark pipeline.
- Used the Talking / Yawning / Normal evaluation set.
- Evaluated Accuracy, Precision, Recall and F1-score.
- Produced confusion matrix and per-class PR curves.
- Analysed temporal-window/class-weighting effects.
- Provided successful and failed qualitative sequences.
- Performed error analysis including talking-vs-yawning confusion.
- Documented limitations and expected-vs-actual gaps.

## Shubham
**Driver Activity Evaluation + Overall Results Comparison**
- Evaluated the MobileNetV3 classifier.
- Used a driver-disjoint evaluation split.
- Evaluated Accuracy, Precision, Recall and F1-score.
- Produced confusion matrix and per-class ROC/PR curves.
- Compared MobileNetV3 against ResNet50/EfficientNet-B0.
- Compared accuracy, parameters, FLOPs and inference speed.
- Analysed qualitative successes/failures including texting-vs-talking confusion.
- Compiled the overall five-module results comparison and computational-cost summary.

## Sohini
**Seat Belt & Phone Evaluation + Common Evaluation Environment and Plotting**
- Evaluated the YOLOv8 seat belt/phone detector.
- Evaluated mAP@50, mAP@50–95, Precision and Recall.
- Compared YOLOv8n, YOLO11n and YOLOv8s configurations.
- Produced PR curves.
- Analysed true positives, missed detections and false detections.
- Investigated small-object, occlusion and seat-belt-colour issues.
- Documented limitations and anomalies.
- Documented the shared GPU/CPU, CUDA, PyTorch/Ultralytics and runtime environment.
- Prepared unified plotting utilities for confusion matrices and ROC/PR curves.

## Ravina
**Smoking & Drinking Evaluation + Consolidation**
- Evaluated the YOLOv8 Smoking/Drinking detector.
- Evaluated mAP@50, mAP@50–95, Precision and Recall.
- Produced PR curves.
- Compared candidate configurations.
- Provided qualitative successful and failed detections.
- Analysed hand-to-mouth ambiguity and phone-vs-cigarette confusion.
- Documented limitations and anomalies.
- Consolidated all member evaluation sections into `Milestone-5-Report.md`.
- Inserted result plots.
- Verified tables and references.
- Performed final proofreading.
- Prepared/merged the M5 presentation.
- Maintained the contribution tracker.
- Prepared the submission checklist.

### M5 review follow-ups recorded in the tracker

The M5 tracker also records technical follow-ups from the M4 review, including:

- Video fatigue: complete training to convergence and improve checkpoint persistence — Kushagra.
- Activity module: investigate the data-balancing paradox and class redundancy — Shubham.
- Smoking/Drinking: reduce false positives through threshold optimization — Ravina.
- Shared distinct unseen test split — all members, with protocol responsibility assigned to Kushagra.
- Visualization consolidation — all owners, with consolidation assigned to Sohini.
- Qualitative runtime examples — all owners, with consolidation assigned to Ravina.
- Metric discrepancy resolution — Shubham, with final check by Ravina.
- Peer-review initials/signatures — all members, tracker maintained by Ravina.

The tracker also records that Ravina was absent from the mandatory M4 review meeting and assigns a score of zero for that review, while also recording subsequent technical follow-up work.

---

# 8. Milestone 6 — Final Integration, Gradio and Lightning ai Deployment

Milestone 6 focused on converting the five independently developed/evaluated modules into one integrated Driver Wellness & Safety Monitoring System.

The milestone included:

- Five-model integration.
- Standardized inputs and outputs.
- Risk Fusion Engine integration.
- Driver Wellness Score.
- Final risk category.
- Gradio interface.
- Lightning ai deployment.
- Deployment testing.
- Documentation and final report consolidation.

## Kushagra — Video Fatigue Integration
- Integrated the final Video Fatigue checkpoint.
- Verified checkpoint loading.
- Connected video preprocessing.
- Verified frame/sequence generation.
- Verified temporal/sliding-window inference.
- Standardized module output.
- Tested normal/safe driving and high-risk/drowsy sequences.
- Tested short, low-quality and insufficient-frame videos.
- Tested invalid/incomplete input.
- Verified risk contribution to the Risk Fusion Engine.
- Prepared M6 integration/testing documentation.
- Supported local and deployment-specific validation.

## Shiwani — Landmark Fatigue Integration and Deployment Support
- Integrated the landmark-based fatigue pipeline.
- Connected face-landmark extraction to final inference.
- Verified uploaded-video compatibility.
- Verified frame-by-frame landmark processing.
- Verified temporal aggregation.
- Validated eye-related features, mouth-related features, head pose, facial landmarks and temporal indicators.
- Checked handling of missing/invalid landmark frames.
- Standardized module output.
- Tested alert/awake, drowsy, head-pose, partial-face and poor-lighting cases.
- Verified risk-fusion integration.
- Supported deployment testing and troubleshooting.
- Investigated deployment-specific MediaPipe/model-loading/runtime issues.

## Shubham — Driver Activity Integration
- Integrated the final Driver Activity model.
- Verified model loading.
- Connected preprocessing and inference.
- Verified class mapping and confidence calculation.
- Standardized the output.
- Supported integrated application testing.
- Maintained Driver Activity integration and verification.

The M6 tracker records **no deployment contribution for Shubham at that stage**; therefore this document does not assign deployment work beyond what the tracker explicitly records.

## Sohini — Seat Belt/Phone Integration, Deployment Coordination and Final Documentation
**M6 Lead**

- Integrated the final Seat Belt & Phone model.
- Verified checkpoint loading.
- Verified YOLO configuration.
- Verified Phone/Seatbelt class mapping.
- Verified confidence thresholds and NMS.
- Verified temporal consensus/stabilization.
- Verified streaming inference.
- Tested Phone Only, Seatbelt Only, Phone & Seatbelt and No Detection scenarios.
- Investigated false positives, false negatives, class errors, confidence issues and temporal instability.
- Verified annotated bounding boxes, labels, confidence and temporal stability.
- Standardized the module output.
- Verified downstream Risk Fusion consumption.
- Verified phone/seatbelt risk mappings.
- Supported Lightning ai validation.
- Modified deployment parameters/configuration.
- Documented deployment limitations.
- Coordinated contribution tracking.
- Coordinated the collection and formatting of M1–M6 contributions.
- Coordinated final consolidated documentation.
- Coordinated consistency checks across reports, presentation, user guide, developer guide and contribution tracker.
- Coordinated final presentation preparation.

## Ravina — Smoking/Drinking Integration and Initial Deployment Documentation
- Integrated the final Smoking/Drinking model.
- Verified model loading, preprocessing, class mapping, confidence and risk mapping.
- Standardized module output.
- Tested normal driving, smoking, drinking, different camera conditions, occlusion and low-quality frames.
- Tested false-positive scenarios.
- Verified Risk Fusion output.
- Prepared the initial Gradio/Lightning ai deployment documentation.
- Documented the initial deployment procedure and configuration.
- Provided deployment guidance to the team.
- Supported Lightning ai testing and troubleshooting.
- Recorded deployment steps and configuration changes.
- Maintained the deployment checklist.
- Supported other members in following the initial deployment process.
- Prepared the M6 report contribution.
- Supported final presentation preparation.

---

# 9. Shared Cross-Milestone Responsibilities

The contribution trackers show that the project was not completed as five isolated modules. Several responsibilities were shared.

### Shared technical responsibilities
- End-to-end integration.
- Input/output compatibility.
- Risk Fusion Engine validation.
- Final Driver Wellness Score verification.
- Failure handling.
- Representative-video testing.
- Review and validation of module outputs.

### Shared documentation responsibilities
- Individual milestone report sections.
- References and work logs.
- Final report review.
- Technical consistency across project documents.
- Final presentation contributions.

### Shared final-system testing
The final system was designed to test scenarios including:

- Normal driving.
- Drowsy driving.
- Phone usage.
- No seatbelt.
- Smoking.
- Drinking.
- Distracted activity.
- Multiple simultaneous risks.
- Poor-quality video.
- Short video.

---

# 10. Final Contribution by Member

## Kushagra — Video Fatigue / Temporal Deep Learning
Kushagra's contribution spans the full technical lifecycle of the Video-Based Fatigue Detection module:

**M1:** Problem definition and project planning.  
**M2:** UTA-RLDD dataset preparation, EDA, preprocessing and split strategy.  
**M3:** CNN-LSTM architecture selection and temporal pipeline design.  
**M4:** CNN-LSTM training, evaluation and checkpointing.  
**M5:** Video fatigue evaluation, error analysis and shared evaluation protocol.  
**M6:** Video fatigue integration, inference validation and end-to-end testing.

**Major cross-cutting contribution:** Shared evaluation protocol.

---

## Shiwani — Landmark Fatigue / Temporal Facial Analysis
Shiwani's contribution spans the full technical lifecycle of the Landmark-Based Fatigue module:

**M1:** Dataset research and understanding.  
**M2:** Landmark temporal-analysis dataset/processed-data responsibility recorded.  
**M3:** EAR/MAR/head-pose feature engineering, temporal architecture and unified I/O specifications.  
**M4:** LSTM training, class balancing, tuning and final report integration.  
**M5:** Landmark evaluation, plots and error analysis.  
**M6:** Landmark integration, feature validation, temporal verification and deployment support.

**Major cross-cutting contribution:** Unified input/output and feature documentation.

---

## Shubham — Driver Activity / Classification
Shubham's contribution spans the full technical lifecycle of the Driver Activity module:

**M1:** Deep-learning and YOLO literature review.  
**M2:** Driver activity dataset/preprocessing/model-readiness responsibility recorded.  
**M3:** MobileNetV3 model selection, architecture and computational comparison, plus M3 report integration.  
**M4:** MobileNetV3 training, tuning, evaluation and computational comparison.  
**M5:** Driver activity evaluation and overall results/computational comparison.  
**M6:** Driver Activity integration and module verification.

**Major cross-cutting contribution:** Overall results comparison and computational-cost summary.

---

## Sohini — Seat Belt & Phone / Detection and Final Consolidation
Sohini's contribution spans the full technical lifecycle of the Seat Belt & Phone Usage module and increasingly broader project coordination:

**M1:** Rule-based and MediaPipe/OpenCV literature review.  
**M2:** Seat Belt/Phone dataset and data-quality/leakage-prevention responsibility recorded.  
**M3:** YOLO model selection and M4 training strategy.  
**M4:** YOLOv8 training, tuning and shared training infrastructure.  
**M5:** Seat Belt/Phone evaluation, shared evaluation environment and unified plotting utilities.  
**M6:** Seat Belt/Phone integration, detection validation, deployment configuration/coordination, contribution tracking and final documentation coordination.

**Major cross-cutting contribution:** M6 documentation/contribution coordination and final consolidated project documentation.

---

## Ravina — Smoking & Drinking / Documentation and Deployment
Ravina's contribution spans the full technical lifecycle of the Smoking & Drinking module and substantial project documentation/presentation work:

**M1:** Evaluation plan and metrics.  
**M2:** Smoking/Drinking dataset, presentation, work log, review and submission-checklist responsibility recorded.  
**M3:** Smoking/Drinking model architecture and model selection, plus presentation/tracker management.  
**M4:** YOLOv8 training/evaluation plus presentation and contribution tracker.  
**M5:** Smoking/Drinking evaluation, error analysis, final report consolidation, presentation and tracker.  
**M6:** Smoking/Drinking integration, initial Lightning ai deployment documentation and deployment support.

**Major cross-cutting contribution:** Report/presentation/tracker consolidation and initial deployment documentation.

---

# 11. Contribution Evidence and Transparency

The milestone trackers support the following conclusions:

1. **Each member had primary technical ownership of one major module or project area.**
2. **Each module progressed through the project lifecycle:** dataset → architecture → training → evaluation → integration.
3. **Cross-functional responsibilities were distributed across the team.**
4. M3 and M4 explicitly document common completion activities for all five module owners.
5. M5 explicitly assigns common evaluation responsibilities in addition to module-specific evaluation.
6. M6 explicitly requires individual M1–M6 contributions to be consolidated into individual technical reports.
7. Integration, testing, documentation and deployment were not intended to be the responsibility of only one person.
8. The final contribution record should be checked against actual notebooks, code commits, reports, model files, test outputs and other evidence before submission.

---

# 12. Recommended Final Submission Table

| Member | Primary Technical Area | M1–M2 Research/Data | M3 Architecture | M4 Training | M5 Evaluation | M6 Integration | Documentation / Presentation |
|---|---|---|---|---|---|---|---|
| **Kushagra** | Video Fatigue | ✓ | ✓ | ✓ | ✓ | ✓ | Evaluation protocol / technical report |
| **Shiwani** | Landmark Fatigue | ✓ | ✓ | ✓ | ✓ | ✓ | I/O/feature documentation, report integration |
| **Shubham** | Driver Activity | ✓ | ✓ | ✓ | ✓ | ✓ | Results comparison / M3 report integration |
| **Sohini** | Seat Belt & Phone | ✓ | ✓ | ✓ | ✓ | ✓ | Training infrastructure, plotting, M6 coordination |
| **Ravina** | Smoking & Drinking | ✓ | ✓ | ✓ | ✓ | ✓ | Reports, presentations, tracker, deployment documentation |

> The checkmarks indicate documented involvement across the milestone trackers. They should not be interpreted as identical amounts of work.

---

# 13. Final Overall Contribution Statement

The project was completed through **distributed technical ownership combined with shared integration, evaluation, documentation and deployment activities**.

Each team member developed and maintained a primary technical area while also contributing to the broader project lifecycle. The project evolved from problem definition and research in M1, through dataset preparation in M2, architecture and system planning in M3, model training in M4, evaluation and error analysis in M5, and finally integration, Gradio interface development, deployment and final documentation in M6.

The final contribution record therefore reflects both:

- **Individual technical ownership**, and
- **Shared project-level responsibilities**.

This is particularly important for M6, where five independently developed models had to operate together through a common inference pipeline, Risk Fusion Engine, Driver Wellness Score and user-facing interface.

---

# 14. Final Verification Note

Before final submission, each team member should review this document against:

- Their M1–M6 reports.
- Training/evaluation notebooks.
- Source-code contributions.
- Git commits where applicable.
- Model/checkpoint artifacts.
- Evaluation plots and results.
- Presentation contributions.
- Deployment evidence.
- Documentation and review records.

Any discrepancy should be corrected before the team signs off on the final contribution statement.

**Prepared for:** Final Project Submission  
**Project:** AI-Powered Driver Wellness & Safety Monitoring System  
**Coverage:** Milestones 1–6


---

# 15. Corrected Milestone 6 Contribution Summary

> This section supersedes the earlier M6 contribution wording. The existing M1–M5 contribution history is retained unchanged. The following M6 details are based on the corrected Milestone 6 Team Contribution Tracker supplied by the team.

## 15.1 Milestone 6 Lead

**Sohini**

M6 focuses on final integration, deployment, documentation, final reporting and submission preparation.

## 15.2 Final M6 Contribution Distribution

| Team Member | Primary M6 responsibility | Additional M6 contribution |
|---|---|---|
| **Kushagra** | Video-Based Fatigue Detection integration | Inference validation, end-to-end integration testing, deployment-specific model validation/support |
| **Shiwani** | Landmark-Based Fatigue Detection integration | Feature/input validation, temporal output verification, deployment support, deployment/runtime troubleshooting |
| **Shubham** | Driver Activity Classification integration | Gradio interface, deployment, Driver Activity verification and deployment-related configuration/support |
| **Sohini** | Seat Belt & Phone Usage Detection integration | Detection validation, deployment support/coordination, documentation coordination, contribution tracking, final consolidated reports and final presentation preparation |
| **Ravina** | Smoking & Drinking Detection integration | Initial Lightning ai deployment documentation, deployment support, deployment checklist/configuration guidance, final presentation preparation |

## 15.3 Shared M6 Responsibility

All five members have a shared responsibility for:

- Integrating and validating their own module within the complete pipeline.
- Verifying input/output compatibility.
- Testing representative driver videos.
- Supporting end-to-end integration testing.
- Identifying module failures and runtime issues.
- Supporting documentation.
- Contributing to the final project report.
- Supporting deployment testing and stabilization where applicable.

## 15.4 Kushagra — Video-Based Fatigue Detection

Kushagra's M6 work centers on integrating the final Video-Based Fatigue Detection module.

Key contributions:

- Final checkpoint integration.
- Checkpoint loading verification.
- Video preprocessing and sequence-generation verification.
- Temporal/sliding-window inference verification.
- Standardized module output.
- Integration testing using normal, drowsy, short, low-quality and invalid inputs.
- Risk Fusion contribution verification.
- M6 technical documentation.

## 15.5 Shiwani — Landmark-Based Fatigue Detection

Shiwani's M6 work centers on Landmark-Based Fatigue Detection integration and validation.

Key contributions:

- Landmark fatigue pipeline integration.
- Face landmark extraction verification.
- Feature/input validation.
- Temporal aggregation verification.
- Handling of missing/invalid landmark frames.
- Standardized module output.
- Integration and robustness testing.
- Risk Fusion verification.
- Deployment support and troubleshooting.
- Deployment-specific testing of MediaPipe/model/runtime behavior.
- M6 documentation.

## 15.6 Shubham — Driver Activity Classification

Shubham's M6 work centers on Driver Activity Classification integration and the Gradio/deployment side of the system.

Key contributions:

- Driver Activity model integration.
- Model loading and preprocessing verification.
- Class mapping and confidence verification.
- Standardized module output.
- Driver Activity documentation.
- Gradio interface work.
- Lightning ai deployment work/support.
- Deployment parameter/configuration changes and deployment troubleshooting.
- Integrated application verification.

> The tracker also records deployment-related work on other platforms; these activities should be described as deployment support/configuration work rather than replacing the final documented Lightning ai deployment path.

## 15.7 Sohini — Seat Belt & Phone Usage Detection

Sohini's M6 responsibility includes both technical module integration and final project coordination.

Key contributions:

- Seat Belt & Phone Usage Detection integration.
- YOLO checkpoint and configuration verification.
- Class mapping verification.
- Confidence-threshold/NMS validation.
- Streaming inference verification.
- Phone/seatbelt scenario validation.
- False-positive/false-negative investigation.
- Annotated-output verification.
- Standardized output verification.
- Risk Fusion verification.
- Lightning ai deployment support and coordination.
- Deployment configuration/parameter modification in the Lightning ai Space.
- Documentation coordination.
- Contribution tracking.
- Consolidation of individual M1–M6 reports.
- Coordination of the final consolidated project report.
- Coordination of the final technical, non-technical, user and developer documentation.
- Final presentation preparation and consistency checking.

## 15.8 Ravina — Smoking & Drinking Detection + Initial Deployment Documentation

Ravina's M6 responsibility includes Smoking/Drinking integration and the initial deployment documentation.

Key contributions:

- Smoking & Drinking model integration.
- Model loading and preprocessing verification.
- Class mapping, confidence and risk mapping verification.
- Module testing under different conditions.
- Risk Fusion validation.
- Initial Lightning ai deployment documentation.
- Initial deployment procedure and configuration guidance.
- Deployment checklist.
- Deployment testing and troubleshooting support.
- Deployment evidence/screenshots where available.
- Support to other members following the initial deployment procedure.
- Individual M6 report contribution.
- Final presentation preparation.

## 15.9 Documentation Contribution

Sohini coordinates the final documentation consolidation, while each member remains responsible for the technical accuracy of their own module sections.

The final documentation process includes:

1. Individual M1–M6 reports.
2. Module-specific technical sections.
3. Integrated architecture and pipeline.
4. Risk Fusion documentation.
5. Gradio interface documentation.
6. Lightning ai deployment documentation.
7. Deployment limitations.
8. Final consolidated project report.
9. Final presentation.
10. Final contribution tracker.

## 15.10 Current M6 Deployment Status

**Deployment platform documented by the corrected M6 tracker:** Lightning ai + Gradio.

**Current status:** Deployment has been initiated and configuration/testing has been performed. 

## 15.11 Overall M1–M6 Contribution Statement

The project progressed from individual module development and evaluation in M1–M5 to integrated application development, deployment and documentation in M6.

```text
M1 → Problem / Dataset / Initial Development
 ↓
M2 → Data Preparation / Analysis
 ↓
M3 → Module Development / Architecture
 ↓
M4 → Training / Optimization
 ↓
M5 → Evaluation / Error Analysis
 ↓
M6 → Integration / Risk Fusion / Gradio / Deployment / Documentation
```

Each member retained primary ownership of their assigned model while participating in shared integration, testing, documentation and deployment activities.

The M6 contribution distribution reflects the corrected tracker and distinguishes **primary module ownership** from **shared project-level responsibilities**.
