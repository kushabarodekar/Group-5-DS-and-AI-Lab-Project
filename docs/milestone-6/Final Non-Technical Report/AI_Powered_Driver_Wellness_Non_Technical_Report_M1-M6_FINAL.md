# AI-Powered Driver Wellness and Safety Monitoring System
## Final Consolidated Non-Technical Project Report — M1–M6

> **Purpose:** Explain the project, its development journey, functionality, results, challenges, impact and future direction for a general audience. Heavy implementation detail is intentionally excluded.
>
> **M6 update:** The project was integrated into a five-model Risk Fusion Engine and prepared for final deployment on Lightning.ai with a Gradio user interface. Lightning.ai is the final deployment platform.

## 1. Executive Summary

Road safety is influenced not only by the road and vehicle, but also by the driver's condition and behaviour. A driver can become unsafe because of fatigue, distraction, phone use, lack of seat-belt use, smoking, drinking, or other activities that take attention away from driving.

The AI-Powered Driver Wellness and Safety Monitoring System was developed as an academic prototype to monitor several visible driver-safety signals together. It combines five specialised capabilities:

1. Video-based fatigue detection
2. Facial-landmark-based temporal fatigue analysis
3. Driver activity classification
4. Seat-belt and phone detection
5. Smoking/drinking detection

The central idea is to move from isolated event detection to a broader Driver Wellness view. Each module observes a different aspect of driver behaviour, and the outputs can be combined by a Risk Fusion Engine.

Across M1–M5, the team progressed from problem definition and literature review through dataset preparation, architecture design, training, optimisation, evaluation and detailed error analysis. By M4, all five modules had trained checkpoints and inference artifacts. M5 then showed that some modules were promising for real-time use while others, especially the video-fatigue model, still require substantial improvement before standalone safety use.

## 2. Background and Motivation

The project began with a narrower focus on driver drowsiness but expanded after considering the broader safety problem. A driver may be awake but distracted, wearing a seat belt but using a phone, or showing several risk indicators at once.

The broader direction created a complete machine-learning lifecycle: problem understanding, dataset preparation, model development, training, evaluation, error analysis and integration planning.

The project is an academic prototype rather than a certified commercial safety product.

## 3. Problem Statement

The core problem is how to automatically identify multiple visible indicators of unsafe driving from camera input and turn them into information meaningful at driver or trip level.

The development scope includes recorded video, uploaded video and local webcam input. The intended practical setting is a dashboard or in-vehicle camera rather than a smartphone-only solution.

## 4. Project Objectives

- Study multiple datasets covering fatigue, distraction, object-level safety and driver activity.
- Understand dataset quality, class distributions, video characteristics, lighting variation and annotation limitations.
- Prepare video, image, temporal-feature and object-detection data.
- Develop temporal fatigue monitoring.
- Use facial landmarks as an additional fatigue signal.
- Detect seat-belt and phone-related safety events.
- Recognise distracting driver activities.
- Detect smoking and drinking behaviour.
- Combine outputs into a driver wellness/risk assessment.
- Generate structured trip information and intended human-readable summaries.
- Evaluate the system using appropriate accuracy, detection, sequence and real-time measures.

## 5. Who the System Is Intended to Help

| Stakeholder | Potential benefit |
|---|---|
| Drivers | Awareness of fatigue, distraction, seat-belt issues and unsafe behaviour |
| Passengers | Indirect benefit through safer driver behaviour |
| Cab owners/agencies | Safety-pattern review |
| Truck/logistics operators | Long-distance fatigue and risk monitoring |
| Bus operators | Passenger-safety support |
| Fleet owners | Wellness scores and trend analysis |
| Transport safety reviewers | Structured logs for recurring safety issues |

## 6. What the System Does

The system can be understood as several specialised observers watching the same driving session:

- **Video fatigue:** looks for fatigue patterns over video.
- **Landmark fatigue:** examines facial and head behaviour over time.
- **Driver activity:** recognises safe driving and distracting activities.
- **Seat belt/phone:** detects seat-belt and phone status.
- **Smoking/drinking:** detects unsafe smoking/drinking behaviour.
- **Risk fusion:** combines module information into a broader wellness/risk view.
- **Reporting:** converts structured trip information into understandable summaries.

## 7. The Five Modules

### 7.1 Video-Based Fatigue Detection

This module looks for fatigue patterns that develop across video frames, such as slow blinking, prolonged eye closure, repeated yawning, head nodding, looking down and reduced facial motion.

A single frame can be misleading: closed eyes can be a normal blink and an open mouth can be talking. Sequence-based modelling is therefore used to understand behaviour over time.

The final M5 evaluation reported **33.55% test accuracy**, **27.39% macro F1**, and severe class collapse, including **3.53% recall for Caution** and **67.09% of High-Risk predictions being classified as Safe**. The module was therefore explicitly judged unsuitable for standalone safety deployment at its current performance.

### 7.2 Landmark-Based Fatigue Detection

This module provides a complementary fatigue signal using interpretable facial/head features. The M2 pipeline used YawDD and extracted:

- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head pitch
- Head yaw
- Head roll

The final model classified 45-frame windows into Normal, Talking and Yawning, which were then mapped to Alert, Mild Fatigue or Drowsy.

The M5 test set contained **1,400 subject-disjoint windows**: 648 Normal, 594 Talking and 158 Yawning.

M5 also investigated threshold selection and label-related errors. A 5% threshold produced **94.4% sensitivity, 91.2% specificity and 92.8% balanced accuracy** in the reported threshold experiment.

The module is lightweight and streaming-compatible, although formal end-to-end latency benchmarking and compression testing remained outstanding.

### 7.3 Driver Activity Classification

This module recognises driver activities:

- Safe Driving
- Texting on Phone
- Talking on Phone
- Turning
- Other Activities

The project compared MobileNetV3, ResNet50 and EfficientNet-B0. Documented baseline accuracy was:

| Model | Accuracy |
|---|---:|
| MobileNetV3 | 89.35% |
| ResNet50 | 92.13% |
| EfficientNet-B0 | 93.98% |

The later test evaluation reported MobileNetV3 at **93.14% test accuracy**, **4.2M parameters** and **12.5 ms inference**. ResNet50 was 92.13% at 45.2 ms, while EfficientNet-B0 was 93.98% at 28.7 ms.

MobileNetV3 was therefore favoured for the intended deployment because it offered the best practical accuracy-speed-efficiency balance rather than simply the highest accuracy.

### 7.4 Seat-Belt and Phone Usage Detection

This module answers two important safety questions:

- Is the driver wearing a seat belt?
- Is a phone being used?

The system uses object detection so that it identifies the location of the relevant object in the image.

The notebook-based evaluation reported:

- **mAP@50:** ~0.953
- **mAP@50–95:** 0.714
- **Precision:** 0.937
- **Recall:** 0.907
- **Inference speed:** approximately 6.8–10.4 ms/frame in the reported evaluation

These figures are **validation results from the notebook**, not a separately held-out test result.

Failure analysis identified:

- Phones disappearing in heavy cabin shadows
- Seat belts washing out under strong sunlight
- Reflections resembling phones
- Arm/seat-belt/phone overlap

The team added inference safeguards such as separate confidence thresholds, overlap suppression, spatial filtering and temporal consensus. Approximately **85% frame stability** was required before confirming seat-belt detection in the reported streaming logic.

### 7.5 Smoking and Drinking Detection

This module detects smoking and drinking behaviour using object detection.

The final M5 evaluation used a held-out test set of **371 images and 445 labelled boxes**, with **214 smoking** and **231 drinking** boxes.

The major result was an important class difference: smoking recall was approximately **0.93**, while drinking recall was approximately **0.67**. Cigarettes were particularly difficult because they are very small.

The model measured approximately:

- **3,011,238 parameters**
- **8.1 GFLOPs at 640×640**
- **4.1 ms inference**
- **~6.9 ms end-to-end latency on Tesla T4**

This shows strong computational efficiency, but also demonstrates that fast inference does not automatically mean equally reliable detection of every behaviour.

## 8. How the Modules Work Together

| Module | Main output | Role |
|---|---|---|
| Driver Activity | Activity category | Immediate distraction/safety signal |
| Seat Belt & Phone | Seat-belt and phone status | Compliance/distraction |
| Smoking & Drinking | Unsafe behaviour | Health/safety signal |
| Video Fatigue | Fatigue level | Accident-risk signal |
| Landmark Fatigue | Drowsiness/behaviour level | Complementary fatigue signal |

The M3 design proposed a weighted wellness score:

- Driver Activity: 25%
- Seat Belt: 15%
- Smoking/Drinking: 10%
- Video Fatigue: 25%
- Landmark Fatigue: 25%

The intended output is a **0–100 wellness score**, module predictions, risk indicators, recommendations and historical trends.

These weights describe the documented architecture/design and should not be interpreted as proof that the complete end-to-end fusion system has already been production-validated.

## 9. Driver Wellness Score

The purpose of the wellness score is to avoid treating one isolated observation as the entire story. A brief phone detection is different from a situation where phone use, fatigue and repeated distraction are all observed together.

The intended states are:

- **Safe:** behaviour appears normal.
- **Caution:** mild fatigue or distraction is detected.
- **High Risk:** strong fatigue, unsafe activity or multiple risk signals are detected.

Trip reports can use trip duration, average wellness score, fatigue events, high-risk events, phone usage, seat-belt violations, smoking/drinking events and time spent in each risk state.

## 10. M1 — Problem Definition and Literature Review

M1 expanded the project from narrow drowsiness detection into broader driver wellness and safety monitoring.

The team reviewed:

- Rule-based facial systems
- Image-based object detection
- Temporal video models
- Driver activity recognition
- Reporting systems

The project also identified real-world difficulties such as normal blinking versus prolonged eye closure, yawning versus talking, glasses/sunglasses, occlusion, lighting and camera-angle variation.

The key M1 outcome was a defined problem, scope, objectives, stakeholder map, candidate datasets and system direction.

## 11. M2 — Dataset Preparation

M2 converted the concept into model-ready data.

The milestone covered:

- Dataset verification
- Dataset ownership/usage information
- Class and metadata analysis
- EDA and quality checks
- Invalid/duplicate sample handling
- Train/validation/test splitting
- Leakage prevention
- Reproducible preprocessing
- Model-ready folder structures

| Module | Dataset/source | Prepared information |
|---|---|---|
| Landmark fatigue | YawDD | EAR, MAR, pitch, yaw, roll temporal features |
| Video fatigue | UTA-RLDD | Fatigue-state video/frame data |
| Driver activity | AUC Distracted Driver Dataset | Activity images |
| Seat belt/phone | DMS dataset | YOLO object-detection data |
| Smoking/drinking | YOLO-format dataset | Object-detection images/labels |

M2 also documented limitations such as controlled environments, underrepresented night/rain/glare/occlusion, near-duplicate frames and incomplete subject identifiers.

## 12. M3 — Model Architecture and End-to-End Design

M3 converted the prepared data into model designs.

| Module | Model/type | Main reason |
|---|---|---|
| Video fatigue | CNN-LSTM / later EfficientNet-B0 + BiLSTM | Temporal behaviour |
| Landmark fatigue | LSTM | Facial feature sequences |
| Driver activity | MobileNetV3 | Lightweight classification |
| Seat belt/phone | YOLOv8n | Fast object detection |
| Smoking/drinking | YOLOv8n | Efficient object detection |

The end-to-end concept was:

**Camera/Video → Five Modules → Risk Fusion → Driver Wellness Score → Driver Report/Dashboard**

The modular architecture allows individual components to be improved without redesigning the entire system.

## 13. M4 — Model Training and Development

M4 followed a common workflow:

**Dataset → Preprocessing → Model Selection → Hyperparameter Optimisation → Training → Validation → Evaluation → Best Checkpoint → Inference Pipeline**

The documented environment primarily used:

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Google Colab
- NVIDIA Tesla T4 GPU

By the end of M4, all five modules had trained checkpoints and supporting artifacts.

The milestone found that transfer learning reduced training time, temporal models captured fatigue-related behaviour, MediaPipe provided compact facial features, YOLO detectors were efficient, and MobileNetV3 provided a strong accuracy-efficiency balance.

## 14. M5 — Evaluation and Analysis

M5 focused on:

- Held-out evaluation where available
- Quantitative metrics
- Qualitative results
- Error analysis
- Robustness
- Operational limitations
- Deployment-readiness considerations

The most important outcome was that the modules did **not** perform equally.

| Module | Representative result | Interpretation |
|---|---|---|
| Video fatigue | 33.55% test accuracy | Not ready for standalone safety use |
| Landmark fatigue | Threshold tuning improved reported operating point | Promising experimental component; further validation required |
| Driver activity | 93.14% test accuracy, 12.5 ms | Strong accuracy-speed balance |
| Seat belt/phone | ~0.953 validation mAP@50 | Strong validation performance; continuous-video robustness matters |
| Smoking/drinking | Smoking recall ~0.93; drinking ~0.67 | Fast but weaker on drinking and small objects |

The project therefore reports the weaknesses rather than hiding them behind a single overall score.

## 15. What the Team Learned from Errors

The major lessons were:

- Fatigue models can confuse visually similar behaviours when labels do not precisely match time windows.
- Landmark fatigue performance can depend strongly on labelling and threshold choices.
- Seat-belt/phone detection is sensitive to shadows, glare, reflections and overlapping arms/objects.
- Smoking/drinking detection is sensitive to small objects and has a weaker drinking class.
- Driver activity classes can be visually similar.
- Continuous video creates flickering and temporal inconsistencies that static-image metrics may not show.

The team responded with threshold tuning, temporal consensus, spatial filtering, overlap suppression, label investigation and proposals for better temporal modelling and data.

## 16. Major Challenges and Responses

| Challenge | Response |
|---|---|
| Different behaviours need different data | Separate datasets and pipelines |
| Class imbalance | Class-aware preparation/training and class-level analysis |
| Lighting/camera variation | Failure analysis and robustness improvements |
| Occlusion/overlap | Filtering and temporal logic |
| Temporal fatigue behaviour | Sequence-based modelling |
| Real-time constraints | Lightweight architectures and latency measurements |
| Leakage risk | Split controls and subject-disjoint evaluation where available |
| Weak performance | Explicit diagnosis and improvement plans |

## 17. What the Project Achieved by M5

- Complete multi-module problem definition
- Prepared datasets for five monitoring areas
- Modular end-to-end architecture
- Trained checkpoints for all five modules
- Inference notebooks/pipelines
- Quantitative and qualitative evaluation
- Detailed failure analysis
- Operational limitations
- Real-time-compatible measurements for relevant components
- Clear next steps for integration and improvement

## 18. What a Future User Would See

A future user would not need to understand the underlying AI models. The intended interface could show:

- Overall wellness score
- Individual module indications
- Current risk level
- Historical trends
- Trip summary
- Fatigue trend
- Distraction trend
- Seat-belt compliance
- Unsafe-behaviour events
- Suggestions for safer driving

The planned reporting layer can transform structured JSON/CSV event information into human-readable trip reports rather than processing raw video with the language model.

## 19. Practical Impact and Potential Applications

Potential applications include:

- Driver awareness and early warning
- Fleet safety monitoring
- Long-distance truck/logistics monitoring
- Bus safety review
- Cab/ride-service safety analysis
- Research and further driver-monitoring development

These are potential applications, not claims that every application has already been commercially deployed.

## 20. Current Limitations and Responsible Use

The project is an academic prototype. It does not:

- Control braking or steering
- Provide medical diagnosis
- Use ECG/EEG or other physiological sensors
- Claim certified commercial safety
- Claim production hardware deployment as part of M1–M6

Other limitations include underrepresented night/rain/glare/occlusion cases, variation between drivers and camera viewpoints, unequal module performance, incomplete end-to-end validation, and privacy/fairness considerations.

## 21. Future Improvements

- Integrate all five models into one real-time pipeline.
- Complete and validate the central Risk Fusion Engine.
- Test longer and more diverse driving sessions.
- Add nighttime, glare, shadow and occlusion examples.
- Improve fatigue labels and temporal modelling.
- Investigate attention-based temporal architectures.
- Apply quantisation/pruning where appropriate.
- Improve confidence aggregation and temporal smoothing.
- Develop a unified dashboard.
- Strengthen privacy, fairness and responsible-deployment evaluation.
- Complete formal end-to-end latency/throughput benchmarking.
- Expand held-out and out-of-distribution evaluation.

## 22. Overall M1 → M6 Journey

| Milestone | Outcome |
|---|---|
| M1 | Problem, scope, objectives, stakeholders, literature and project direction |
| M2 | Dataset identification, cleaning, preprocessing and model-ready structures |
| M3 | Five-module architecture and end-to-end risk-fusion design |
| M4 | Training, tuning, checkpoints and inference artifacts |
| M5 | Evaluation, error analysis, limitations and improvement directions |

The central story is that the team did not jump directly from an idea to a model. Each milestone added another layer of evidence and understanding.

## 23. M6 — Integration, Final Application and Deployment

M6 moved the project from a collection of separately developed models toward a single driver-wellness application.

The five capabilities were brought together:

- Video-based fatigue detection
- Landmark-based fatigue detection
- Driver activity classification
- Seat-belt and phone detection
- Smoking/drinking detection

A central **Risk Fusion Engine** combines information from these modules so that the application can present a broader view of driver risk rather than five unrelated model outputs.

### 23.1 Final User Experience

| Mode | What the user does | What the system provides |
|---|---|---|
| Recorded Video | Upload a short driving video and analyse it | Annotated video and driver-risk/session information |
| Live Webcam | Start the webcam and stream the driver's view | Live annotated feedback and a final session summary |

Recorded-video analysis is the preferred mode for demonstrations because it is easier to reproduce and places less demand on the computing environment.

### 23.2 Final Deployment Platform

The final deployment platform selected for M6 is **Lightning.ai**.

The application runs as a Gradio web interface inside a Lightning.ai Studio. GPU-backed inference is used because several AI models participate in the same application.

The final deployment flow is:

**Driver Video/Camera → Five AI Observers → Risk Fusion → Wellness/Risk Result → Gradio Interface**

The project therefore progressed beyond model training into an integrated application and deployment workflow.

### 23.3 What the User Sees

A user does not need to understand the individual AI architectures. The application is intended to provide:

- An annotated view of the driver's video
- Current driver-safety indications
- A combined risk/wellness score
- Risk-level information
- Module-level observations
- A session summary

The system converts complex model outputs into information that is easier for a human to understand.

### 23.4 Final Risk-Fusion Concept

The final system combines risk information produced by the active modules rather than treating one isolated detection as the complete assessment.

In simple terms:

1. Each AI module identifies a behaviour or condition.
2. The system considers the significance and confidence of that observation.
3. Active risks are combined.
4. The combined result is converted into an overall score.
5. Short-term smoothing can reduce unstable frame-to-frame changes.
6. The application presents the resulting driver-risk state.

### 23.5 M6 Deployment Deliverables

The final deployment package contains:

- `app.py` — Gradio application
- `wellness_core.py` — integrated model and risk-fusion logic
- `requirements.txt` — software dependencies
- `README.md` — setup and usage documentation
- `models/` — trained model checkpoints and supporting files

The deployment documentation covers Lightning.ai Studio preparation, dependency installation, model setup, GPU selection, application launch and public Gradio sharing.

### 23.6 M6 Deployment Status

| Area | M6 status |
|---|---|
| Five specialised models | Integrated |
| Risk Fusion Engine | Integrated |
| Gradio application | Implemented |
| Recorded-video workflow | Implemented |
| Live-webcam workflow | Implemented |
| Deployment documentation | Prepared |
| Final deployment platform | **Lightning.ai** |
| GPU-backed inference | Supported |
| Public Gradio sharing | Supported |

The earlier Hugging Face deployment plan was superseded. Lightning.ai is the platform that should be referenced in the final project documentation.

### 23.7 Team Contribution During M6

M6 was a team-level integration and deployment milestone. The documented work included model integration, common risk-fusion outputs, Gradio interface development, Lightning.ai deployment preparation/testing, model/configuration setup and final documentation.

Individual ownership is captured separately in the final contribution summary.

### 23.8 What M6 Added to the Overall Journey

| Milestone | Main outcome |
|---|---|
| M1 | Defined the problem and project direction |
| M2 | Prepared and analysed the datasets |
| M3 | Designed the five-module system |
| M4 | Trained models and produced checkpoints |
| M5 | Evaluated models and analysed weaknesses |
| **M6** | **Integrated the models, built the final application and deployed it through Lightning.ai** |

M6 completed the transition from **individual model development** to a **single integrated driver-wellness application**.

## 24. Conclusion

The AI-Powered Driver Wellness and Safety Monitoring System demonstrates how several AI-based vision capabilities can be combined around one practical road-safety problem.

M1 established the problem and direction. M2 created the data foundation. M3 translated the requirements into a modular architecture. M4 produced trained models and inference artifacts. M5 provided the reality check by showing both what works and what still needs improvement.

The result is more than a collection of individual models: it is a structured foundation for an integrated driver-wellness monitoring system, together with a clear understanding of its current strengths, weaknesses and next steps.

**M6 completed the integration and final deployment stage. The five-model system was brought together through a common Risk Fusion Engine and deployed through Lightning.ai using Gradio.**

## Appendix A — Project at a Glance

| Item | Summary |
|---|---|
| Project type | Academic AI/computer-vision prototype |
| Main input | Driver video/camera imagery |
| Monitoring areas | Fatigue, facial fatigue, activity, seat belt/phone, smoking/drinking |
| Architecture | Five specialised modules connected through a planned risk-fusion layer |
| Main outcome | Driver Wellness and Safety assessment |
| Development covered | M1 problem definition through M6 integration and deployment |
| Current document status | Final consolidated M1–M6 non-technical report |
| M6 status | Integrated five-model application and final Lightning.ai deployment |
