# AI-Powered Driver Wellness and Safety Monitoring System **Milestone 1**

---

## 1. Introduction

Driver safety is not limited to detecting whether a driver is sleepy or not. In real-world driving, driver safety depends on multiple factors such as fatigue, distraction, unsafe behavior, seat belt compliance, phone usage, smoking or drinking while driving, head movement, gaze direction, and long-duration driving behavior. A driver may look normal in a single image frame, but their actual condition can be understood more accurately by observing continuous video behavior over time. Fatigue is especially a time-based condition. It usually appears through patterns such as slow blinking, prolonged eye closure, repeated yawning, head nodding, unstable posture, or reduced attention. Therefore, a system that checks only a single image or one frame at a time may not be reliable enough for driver monitoring. Our updated project direction is titled **AI-Powered Driver Wellness and Safety Monitoring System**. 

The system aims to monitor the overall driver state using video-based deep learning, temporal modeling, object detection, landmark-based feature extraction, and driver behavior analysis. After the Milestone 1 discussion and TA suggestions, we explored a broader driver-monitoring direction that focuses not only on drowsiness, but also on overall driver wellness and safety. The proposed system will process live or recorded driver video and detect fatigue-related behavior, distraction, seat belt usage, phone usage, smoking or drinking activity, head pose, and gaze-away behavior. The outputs from these modules will be combined into a driver wellness score. The system will also generate trip-level or weekly/monthly driver reports using structured event logs. This makes the project more useful for drivers, fleet owners, cab agencies, logistics companies, and transport operators.

---

## 2. Problem Statement

The problem is to design and implement a real-time, video-based driver wellness and safety monitoring system that can identify fatigue, distraction, unsafe driving behavior, and safety compliance using multiple machine learning and deep learning models. Our initial exploration started with frame-level drowsiness cues using YOLO and simple temporal checks. After the TA's suggestions, we explored the problem more deeply and understood that driver fatigue needs stronger video-based temporal modeling rather than only frame-level detection. Driver fatigue cannot be reliably identified using only isolated images. For example, a normal blink, talking, singing, laughing, or briefly looking down can cause false alerts if the system depends only on a single frame or hardcoded conditions. The proposed system will process continuous driver video and learn temporal patterns. The fatigue detection module will use video sequences rather than static images. Models such as CNN-LSTM, CNN-GRU, Temporal Convolutional Networks, or lightweight 3D CNNs will be explored to learn how fatigue develops across time. MediaPipe/OpenCV will be used only as auxiliary tools for extracting interpretable features such as EAR, MAR, head pose, and gaze-related signals. These features will be passed to trainable temporal models instead of using fixed static threshinitials as the final decision logic. YOLO will be used for object-level safety tasks such as seat belt detection, phone detection, smoking detection, and drinking detection. The final driver state will be generated using model fusion and a wellness scoring layer. The system will classify driver condition as **Safe**, **Caution**, or **High Risk**.

---

## 3. Motivation

The motivation behind the updated project is to address driver safety in a broader and more realistic way. Fatigue is a major road safety issue, especially during night driving, long-distance travel, commercial transport, cab driving, and logistics operations. However, drowsiness is not the only reason a driver may become unsafe. A driver may also be distracted by phone usage, may not be wearing a seat belt, may be smoking or drinking while driving, or may frequently look away from the road. A complete driver monitoring system should consider these different factors together. This expanded direction also better matches the learning goals of a machine learning and deep learning course. After reviewing the TA feedback, we explored a more complete ML workflow that includes dataset collection, EDA, video preprocessing, sequence creation, feature extraction, model training, model evaluation, and local deployment. This gives the team a chance to work through the complete ML project lifecycle. The project also has strong practical relevance. Fleet owners can use such a system to monitor driver wellness and risky behavior. Cab agencies can use it for passenger safety. Logistics companies can use it for long-distance transport monitoring. Individual drivers can use it as an early warning system. Transport operators can use generated trip reports to identify safety trends.

---

## 4. Scope and Boundaries

The current scope is to build an academic prototype of a driver wellness and safety monitoring system using video input. For development and demonstration, the system can use recorded video datasets, uploaded videos, and local webcam input. The practical deployment idea is a dashboard camera or embedded in-vehicle camera rather than a smartphone-only solution. The system will focus on software-based detection and analysis.

### What this project covers

- Video-based fatigue detection using temporal deep learning models.
- Landmark-based temporal feature extraction using MediaPipe/OpenCV.
- Driver distraction and activity classification.
- Seat belt detection using object detection.
- Phone, smoking, and drinking detection using object detection.
- Head pose and gaze-away monitoring.
- Driver wellness score calculation.
- Trip-level report generation using structured logs.
- Local dashboard or OpenCV-based demonstration.
- Dataset collection, EDA, preprocessing, sequence creation, model training, and evaluation.

### What this project does not cover

- Real-car mechanical braking integration.
- Automatic steering or vehicle control.
- Certified commercial driver safety product.
- Production hardware deployment.
- Cloud fleet management platform.
- Full mobile application development.
- Medical diagnosis of the driver.
- Use of ECG, EEG, or other physiological sensors.
  The project is strictly video-based and computer-vision-based. It should be treated as an academic prototype and not a certified safety system.

---

## 5. Stakehinitialers

| Stakehinitialer                                                                                                                                          | Relevance / Benefit                                                                 |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Drivers**                                                                                                                                        | Receive warnings about fatigue, distraction, seat belt issues, and unsafe behavior. |
| **Passengers**                                                                                                                                     | Benefit indirectly from safer driving behavior and initial warnings.                |
| **Cab Owners / Cab Agencies**                                                                                                                      | Can monitor safety patterns during customer rides and night operations.             |
| **Truck Agencies / Logistics Companies**                                                                                                           | Can track fatigue and risky driver behavior during long-distance transport.         |
| **Bus Operators**                                                                                                                                  | Can improve passenger safety during long routes and night travel.                   |
| **Fleet Owners**                                                                                                                                   | Can use wellness scores and trip reports to identify risky driving trends.          |
| **Transport Safety Reviewers**                                                                                                                     | Can use system logs to understand repeated driver safety issues.                    |
| The direct users are drivers and vehicle operators. The indirect beneficiaries are passengers, fleet owners, agencies, and road safety stakehinitialers. |                                                                                     |

---

## 6. Project Objectives

The project objectives are:

1. **Collect and study multiple datasets** related to video-based fatigue, distracted driving, object-level safety, and driver activity monitoring.
2. **Perform EDA** on selected datasets to understand class distribution, video duration, frame quality, lighting variation, annotation types, and dataset limitations.
3. **Preprocess video datasets** by extracting frames, face crops, upper-body crops, and fixed-length video sequences.
4. **Develop a temporal fatigue detection module** using CNN-LSTM, CNN-GRU, TCN, or lightweight 3D CNN.
5. **Use MediaPipe/OpenCV as auxiliary feature extractors** for EAR, MAR, head pose, and gaze-related features.
6. **Train supporting object detection modules** for seat belt, phone, smoking, and drinking detection using YOLO-based models.
7. **Train or fine-tune a driver distraction classifier** using lightweight CNN models such as MobileNetV3, EfficientNet-B0, or ResNet18.
8. **Combine model outputs** into a driver wellness score using a fusion layer or scoring logic.
9. **Generate trip reports** using structured logs and an LLM-based report generation module.
10. **Evaluate the system** using accuracy, precision, recall, F1-score, mAP, FPS, false alarm rate, and sequence-level classification metrics.

---

## 7. Dataset Collection and Understanding

Based on the expanded project direction, more than one dataset is required because the system is not limited to only drowsiness detection. Different features require different types of data. For fatigue detection, video datasets are needed. For seat belt and phone detection, object detection datasets are useful. For distracted driving, activity classification datasets are required. For trip reports, no separate raw video dataset is needed; structured logs generated by the system can be used.

### 7.1 Planned Dataset Sources

| Feature Area                                                                                               | Dataset Type                        | Example Dataset / Source                        | Purpose                                                                    |
| ---------------------------------------------------------------------------------------------------------- | ----------------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------- |
| Video-based fatigue detection                                                                              | Driver video dataset                | NTHU Driver Drowsiness Detection Dataset        | Learn drowsiness and fatigue patterns from video sequences.                |
| Yawning and fatigue behavior                                                                               | Driver yawning video dataset        | YawDD Dataset                                   | Study continuous yawning and mouth movement patterns.                      |
| Driver distraction                                                                                         | Driver activity image/video dataset | State Farm Distracted Driver Dataset            | Classify activities such as texting, drinking, reaching, and safe driving. |
| Fine-grained driver action recognition                                                                     | Multi-modal/video dataset           | Drive&Act Dataset                               | Understand driver activities and action recognition.                       |
| Driver monitoring                                                                                          | Video / multi-modal dataset         | DMD Driver Monitoring Dataset                   | Study distraction, gaze, drowsiness, and driver monitoring scenarios.      |
| Seat belt detection                                                                                        | Object detection dataset            | Seat belt datasets from Kaggle / Roboflow       | Detect seat belt presence or absence.                                      |
| Phone/smoking/drinking detection                                                                           | Object detection dataset            | Driver behavior datasets from Roboflow / Kaggle | Detect unsafe objects or activities.                                       |
| Trip report generation                                                                                     | Structured logs                     | Generated by our system                         | Generate driver wellness and safety summaries.                             |
| The final dataset selection will depend on access, size, license, format, and feasibility on Google Colab. |                                     |                                                 |                                                                            |

### 7.2 Why Video Datasets Are Needed

Fatigue is temporal. A single frame can show closed eyes, but it cannot confirm whether the driver is blinking normally or becoming drowsy. A single frame can show an open mouth, but it cannot confirm whether the driver is yawning, talking, laughing, or singing. Video datasets allow the model to observe behavior across time. This helps capture slow blinking, repeated yawning, head nodding, and long eye closure patterns. Therefore, video datasets are essential for the updated fatigue detection module.

### 7.3 Dataset EDA Plan

The following EDA steps will be performed:

- Count number of videos and images in each dataset.
- Identify available classes and labels.
- Check class distribution and imbalance.
- Check video duration and frame rate.
- Extract sample frames from each class.
- Compare lighting conditions such as day, night, low light, and backlight.
- Check driver variations such as eyewear, face angle, and occlusion.
- Check annotation quality and missing labels.
- Identify duplicate, blurry, or unusable samples.
- Understand whether the dataset supports train/validation/test splitting.
  EDA will help decide which datasets are suitable for final training and which are only useful for testing or comparison.

### 7.4 Preprocessing Plan

The preprocessing pipeline will include:

- Reading videos using OpenCV.
- Extracting frames at selected FPS.
- Detecting and cropping face or upper-body region.
- Resizing frames to a fixed resolution.
- Normalizing pixel values.
- Creating fixed-length clips or frame sequences.
- Extracting MediaPipe features per frame.
- Saving processed clips and features in structured finitialers.
- Creating train/validation/test splits.
- Handling class imbalance using sampling or augmentation.
  This preprocessing step is important because raw videos cannot be directly passed into most models without conversion into a consistent format.

### 7.5 Sequence Creation

Instead of treating each frame independently, the system will create fixed-length sequences. For example:

```text
Video → frames → sequence of 16 / 32 / 64 frames → temporal model
```

Each sequence will represent a short time window of driver behavior. The model will learn whether that time window shows alert driving, fatigue risk, or drowsiness. This avoids depending on a static rule-based approach. The sequence length will be treated as a model input design choice, not as a hardcoded drowsiness rule.

### 7.6 Dataset Limitations

The datasets may have some limitations:

- Some datasets may be large and difficult to train fully on Colab.
- Some datasets may have licensing or access restrictions.
- Some datasets may have image data but not video data.
- Some video datasets may have limited subjects.
- Real-world conditions such as rain, night driving, or heavy occlusion may still be underrepresented.
- Object detection datasets may have different annotation formats.
- Driver behavior datasets may not perfectly match our dashboard-camera view.
  These limitations will be documented during Milestone 2.

---

## 8. Literature Review and Existing Solutions

Existing driver monitoring systems can be divided into several categories:

1. Rule-based facial landmark systems.
2. Image-based object detection systems.
3. Video-based temporal deep learning systems.
4. Driver activity recognition systems.
5. Driver wellness and reporting systems.
   Rule-based systems are fast and interpretable but depend heavily on threshinitials. Image-based systems such as YOLO are good for object detection but do not understand temporal behavior by themselves. Video-based deep learning systems can learn motion and behavior patterns across frames. Driver activity recognition systems can identify unsafe actions like texting, drinking, or reaching behind. Reporting systems convert detection outputs into useful summaries for drivers or fleet owners. The proposed project combines these ideas into a multi-module system.

---

## 9. Video-Based Fatigue Detection

Video-based fatigue detection will be the core deep learning component of the proposed project. The model will take video sequences as input instead of isolated images. The system will try to learn patterns such as:

- Slow blinking.
- Prolonged eye closure.
- Repeated yawning.
- Head nodding.
- Looking down for long periods.
- Reduced facial motion.
- Fatigue-related posture changes.

### 9.1 Why Frame-Level Detection Is Not Enough

A single frame can be misleading. Closed eyes may be a normal blink. An open mouth may be talking or yawning. Looking down may be checking the dashboard. Therefore, the model needs to understand time-based behavior. Temporal models can observe how the driver’s face and posture change across multiple frames.

### 9.2 Possible Models

| Model                                                                                                                                                                               | Input                                    | Use Case                                                |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------- |
| **CNN-LSTM**                                                                                                                                                                  | Frame sequence                           | CNN extracts frame features, LSTM learns time behavior. |
| **CNN-GRU**                                                                                                                                                                   | Frame sequence                           | Similar to LSTM but usually lighter and faster.         |
| **TCN**                                                                                                                                                                       | Time-series features or frame embeddings | Captures temporal patterns efficiently.                 |
| **Lightweight 3D CNN**                                                                                                                                                        | Short video clips                        | Learns spatial and temporal features together.          |
| **CNN + Attention**                                                                                                                                                           | Frame sequence                           | Focuses on important frames in a sequence.              |
| For the first implementation, CNN-LSTM or CNN-GRU is the most feasible option on Google Colab. A lightweight 3D CNN can be explored if dataset size and compute resources allow it. |                                          |                                                         |

---

## 10. Landmark-Based Temporal Feature Analysis

MediaPipe and OpenCV will still be useful, but only as feature extraction tools. They will not be used as the final decision-making system.

### 10.1 Features Extracted

The following features can be extracted from each video frame:

| Feature                    | Meaning                                                    |
| -------------------------- | ---------------------------------------------------------- |
| **EAR**              | Eye Aspect Ratio for eye openness.                         |
| **MAR**              | Mouth Aspect Ratio for mouth opening.                      |
| **Head Pitch**       | Up/down head movement.                                     |
| **Head Yaw**         | Left/right head movement.                                  |
| **Head Roll**        | Side tilt of the head.                                     |
| **Blink Pattern**    | Eye closure and reopening behavior over time.              |
| **Gaze-Away Signal** | Whether the driver is looking away from forward direction. |

### 10.2 Temporal Use of Features

The extracted values will be converted into time-series sequences. For example:

```text
Frame 1: EAR, MAR, pitch, yaw, roll
Frame 2: EAR, MAR, pitch, yaw, roll
Frame 3: EAR, MAR, pitch, yaw, roll
...
```

This sequence can be passed to LSTM, GRU, or TCN models. This makes the system learn patterns instead of using only fixed threshinitials.

### 10.3 Role of MediaPipe

MediaPipe will support explainability. For example, if the model predicts fatigue, we can inspect whether EAR was low over time, MAR increased repeatedly, or head pitch changed frequently. This provides interpretable support for the final prediction.

---

## 11. Driver Distraction and Activity Classification

Driver distraction is another important part of driver wellness. A driver may not be sleepy but may still be unsafe. Examples include texting, talking on phone, drinking, reaching behind, operating radio, or talking to a passenger.

### 11.1 Activity Classes

Possible activity classes include:

- Safe driving.
- Texting with right hand.
- Texting with left hand.
- Talking on phone.
- Drinking.
- Reaching behind.
- Operating dashboard or radio.
- Talking to passenger.
- Hair or makeup related distraction.

### 11.2 Possible Models

| Model                                                                                                                                  | Reason for Use                                     |
| -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **MobileNetV3**                                                                                                                  | Lightweight and suitable for local deployment.     |
| **EfficientNet-B0**                                                                                                              | Good accuracy with manageable size.                |
| **ResNet18**                                                                                                                     | Simple and reliable baseline CNN.                  |
| **CNN-LSTM**                                                                                                                     | Useful if video/action sequence data is available. |
| The first baseline can be an image classifier. If video data is available, the model can be upgraded to temporal activity recognition. |                                                    |

---

## 12. Object-Level Safety Detection Modules

YOLO will still be used, but not as the main fatigue detector. YOLO is suitable for visible object detection tasks.

### 12.1 Seat Belt Detection

Seat belt usage is an important safety compliance feature. The system will detect whether the driver is wearing a seat belt. Possible labels:

- Seat belt detected.
- Seat belt missing.
- Person with seat belt.
- Person without seat belt.
  YOLOv8n or YOLO11n can be used for this task.

### 12.2 Phone Usage Detection

Phone usage while driving is a major distraction. The system can detect a mobile phone near the driver. YOLO can identify visible phone objects in the driver cabin.

### 12.3 Smoking and Drinking Detection

The system can also detect unsafe objects or behaviors such as:

- Cigarette.
- Smoking gesture.
- Bottle.
- Cup.
- Drinking action.
  These detections will contribute to the overall driver wellness score.

### 12.4 Why YOLO Is Suitable Here

YOLO is appropriate for object detection tasks because these features are visible in individual frames. However, YOLO will not be used alone to decide fatigue. Fatigue detection will remain video-sequence based.

---

## 13. Proposed System Architecture

The proposed system uses a multi-branch architecture. The main branch handles video-based fatigue detection. The auxiliary branch handles landmark-based temporal features. Additional modules handle object-level safety and driver activity classification.

### 13.1 Architecture Diagram

```text
Input Source
Live Webcam / Recorded Driver Video
        |
        v
Video Preprocessing
- Frame extraction
- Face crop extraction
- Upper-body crop extraction
- Frame resizing
- Normalization
- Fixed-length sequence creation
        |
        v
Multi-Branch Analysis
        |
        +------------------------------------------------------+
        |                                                      |
        v                                                      v
Visual Temporal Branch                              Landmark Temporal Branch
Input: Face/driver video sequence                   Input: EAR, MAR, head pose, gaze features
Model: CNN-LSTM / CNN-GRU / TCN / 3D CNN            Model: LSTM / GRU / TCN
Purpose: Learn fatigue patterns from video          Purpose: Learn interpretable temporal signals
        |                                                      |
        +----------------------------+-------------------------+
                                     |
                                     v
                              Fusion Layer
              Combines visual features and landmark features
                                     |
                                     v
                         Driver State Classification
                       Alert / Fatigue Risk / Drowsy
                                     |
                                     v
Additional Safety Modules
- Seat belt detection using YOLO
- Phone detection using YOLO
- Smoking/drinking detection using YOLO
- Driver activity classification using CNN/EfficientNet/MobileNet
                                     |
                                     v
Driver Wellness Score Calculation
Safe / Caution / High Risk
                                     |
                                     v
Trip Report and Dashboard
- Event logs
- Weekly/monthly health summary
- Driver wellness score
- LLM-generated trip report
```

### 13.2 Improvement from Initial Exploration

| Initial Exploration                 | Expanded Project Direction                           |
| ----------------------------------- | ---------------------------------------------------- |
| Frame-level YOLO detection          | Video-sequence temporal modeling                     |
| Simple static frame-count logic     | Model learns temporal fatigue patterns               |
| Mainly drowsiness-focused detection | Overall driver wellness and safety monitoring        |
| MediaPipe as main logic             | MediaPipe as auxiliary feature extractor             |
| Limited image-dataset focus         | Multiple video, image, and object detection datasets |
| No reporting layer                  | Trip report and wellness dashboard included          |

---

## 14. Driver Wellness Score

The final system will combine multiple model outputs into a driver wellness score. This score will represent the overall safety state of the driver.

### 14.1 Inputs to Wellness Score

| Input Module                    | Example Signal                                    |
| ------------------------------- | ------------------------------------------------- |
| Fatigue model                   | Alert / Fatigue Risk / Drowsy                     |
| Landmark model                  | Eye closure pattern, MAR trend, head pose pattern |
| Distraction classifier          | Texting, phone call, drinking, reaching           |
| Seat belt detector              | Seat belt detected or missing                     |
| Phone/smoking/drinking detector | Unsafe object detected                            |
| Gaze/head pose module           | Looking forward or away                           |

### 14.2 Output Levels

| Output Level                                                            | Meaning                                                                 |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Safe**                                                          | Driver behavior appears normal.                                         |
| **Caution**                                                       | Mild fatigue or distraction is detected.                                |
| **High Risk**                                                     | Strong fatigue, unsafe activity, or multiple risk signals are detected. |
| The wellness score can be shown on a dashboard and stored in trip logs. |                                                                         |

---

## 15. Trip Report Generation

Trip report generation is an add-on intelligence layer. It will not replace the ML models. The ML models will detect fatigue, distraction, object-level safety events, and wellness score. The report module will convert structured logs into readable summaries.

### 15.1 Report Inputs

The report generator can use structured data such as:

- Trip duration.
- Average wellness score.
- Number of fatigue events.
- Number of high-risk events.
- Phone usage count.
- Seat belt violations.
- Smoking or drinking events.
- Longest fatigue event.
- Time spent in Safe, Caution, and High Risk states.

### 15.2 LLM Use

An LLM such as Llama 3.3 70B can be used through an API for report generation. For local demo, a smaller model can be used. The LLM will not process raw video. It will process structured JSON or CSV summaries and generate human-readable trip reports.

### 15.3 Example Report Output

The report can include:

- Trip summary.
- Driver wellness score.
- Safety risk summary.
- Fatigue trend.
- Distraction trend.
- Seat belt compliance.
- Weekly or monthly driver wellness summary.
- Suggestions for safer driving.

---

## 16. Evaluation Plan and Metrics

The proposed system requires evaluation at different levels. Each module will have its own metrics, and the complete system will also be evaluated.

### 16.1 Fatigue Detection Metrics

| Metric                                                                                                                                                                        | Purpose                                           |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Accuracy                                                                                                                                                                      | Overall correct classification of driver state.   |
| Precision                                                                                                                                                                     | Correctness of predicted fatigue alerts.          |
| Recall                                                                                                                                                                        | Ability to catch actual fatigue events.           |
| F1-score                                                                                                                                                                      | Balance between precision and recall.             |
| Confusion Matrix                                                                                                                                                              | Understand class-level errors.                    |
| Sequence-level accuracy                                                                                                                                                       | Measures prediction correctness over video clips. |
| Recall is especially important because missing actual fatigue can be dangerous. Precision is also important because too many false alerts can make drivers ignore the system. |                                                   |

### 16.2 Object Detection Metrics

For YOLO-based modules, metrics include:

- mAP, or Mean Average Precision.
- Precision.
- Recall.
- Class-wise detection accuracy.
- Inference FPS.
  mAP will be used to evaluate seat belt, phone, smoking, and drinking detection.

### 16.3 Real-Time Performance Metrics

The real-time system will be evaluated using:

- FPS.
- Latency.
- Model inference time.
- System response time.
- Local webcam demo performance.
  A system with high accuracy but low FPS may not be useful for real-time monitoring.

### 16.4 False Alarm Analysis

False alarms will be analyzed carefully. Examples of possible false alarms include:

- Talking detected as yawning.
- Normal blink detected as fatigue.
- Looking at mirror detected as distraction.
- Drinking water detected as unsafe behavior depending on context.
  The temporal model and fusion logic should reduce unnecessary warnings.

---

## 17. Expected Output and Testing Scenarios

The final system is expected to produce:

- Driver fatigue state.
- Driver distraction state.
- Seat belt status.
- Phone usage alert.
- Smoking or drinking alert.
- Head pose or gaze-away status.
- Driver wellness score.
- Event log.
- Trip report.
- Dashboard summary.

### 17.1 Testing Scenarios

Testing should include:

- Bright light.
- Low light.
- Night-like conditions.
- Backlight.
- Different camera angles.
- Driver with glasses.
- Driver with sunglasses.
- Normal blinking.
- Prolonged eye closure.
- Yawning.
- Talking.
- Laughing.
- Looking away briefly.
- Phone usage.
- Drinking.
- Seat belt present and missing.
  The system should be tested on both recorded videos and local webcam input.

### 17.2 Local Deployment Plan

The local demo can be implemented using:

- Python.
- OpenCV.
- MediaPipe.
- PyTorch or TensorFlow/Keras.
- Ultralytics YOLO.
- Streamlit or simple OpenCV GUI.
  Google Colab will be used for dataset processing and model training. The trained models will be downloaded and used locally for demonstration.

---

## 18. Team Member Declaration and Initials

We confirm that the updated project direction has been discussed and refined based on the Milestone 1 review feedback and TA suggestions. Instead of official handwritten signatures, initials are used because the repository is public.

| Sr. No. | Team Member Name | Responsibility                                                   | Review Initials |
| ------: | ---------------- | ---------------------------------------------------------------- | --------------- |
|       1 | Kushagra         | Updated scope, architecture, documentation, integration planning | KB              |
|       2 | Shiwani          | Dataset collection, dataset EDA, video data preparation          | ST              |
|       3 | Sohini           | Landmark features, metrics validation, documentation review      | SS              |
|       4 | Shubham          | Deep learning model design, temporal pipeline, YOLO modules      | ST              |
|       5 | Ravina           | Evaluation plan, testing scenarios, reporting review             | R               |

---

## 19. References

1. NTHU Driver Drowsiness Detection Dataset.
2. YawDD: Yawning Detection Dataset.
3. State Farm Distracted Driver Detection Dataset.
4. Drive&Act Driver Activity Recognition Dataset.
5. DMD: Driver Monitoring Dataset.
6. MediaPipe Face Mesh Documentation.
7. OpenCV Documentation.
8. Ultralytics YOLO Documentation.
9. Research papers on CNN-LSTM, GRU, TCN, and 3D CNN for video sequence classification.
10. Research references on Eye Aspect Ratio, Mouth Aspect Ratio, PERCLOS, and head pose estimation.
11. Llama / open-source LLM documentation for report generation from structured logs.
