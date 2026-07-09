# AI-Powered Driver Wellness and Safety Monitoring System

## Overview

This project is an academic prototype for a video-based **Driver Wellness and Safety Monitoring System**. The system is designed to monitor driver fatigue, distraction, seat belt usage, phone usage, smoking, drinking, head movement, gaze behavior, and other unsafe driving indicators.

The project started with drowsiness detection, but after Milestone 1 feedback, the scope was expanded into a broader driver monitoring system. Instead of relying only on frame-by-frame detection or hardcoded rules, the updated approach focuses on dataset preparation, temporal modeling, object detection, landmark-based feature extraction, and multi-module driver wellness analysis.

---

## Project Goal

The goal is to build a system that can analyze driver behavior using computer vision and deep learning, and classify the driver state into:

- **Safe**
- **Caution**
- **High Risk**

The final system is expected to combine outputs from multiple modules into a driver wellness score and generate alerts or reports based on detected risk events.

---

## Key Features

- Video-based fatigue detection using temporal models
- Landmark-based temporal feature extraction using EAR, MAR, head pose, and gaze signals
- Driver distraction and activity classification
- Seat belt detection
- Phone usage detection
- Smoking and drinking detection
- Dataset EDA and preprocessing pipelines
- Train/validation/test split with leakage prevention
- Processed dataset structures for Milestone 3 model training
- Hosted processed dataset links
- Report and presentation documentation

---

## System Modules

| Module | Description | Planned Model Type |
|---|---|---|
| Video-Based Fatigue Detection | Detects fatigue patterns from driver videos | CNN-LSTM, CNN-GRU, TCN, lightweight 3D CNN |
| Landmark-Based Temporal Analysis | Extracts EAR, MAR, head pose, gaze, and facial movement features | LSTM / GRU / TCN |
| Driver Distraction Classification | Classifies activities such as phone use, texting, turning, or safe driving | ResNet, MobileNetV3, EfficientNet |
| Seat Belt + Phone Detection | Detects seat belt and phone usage from driver images | YOLOv8 / YOLO11 |
| Smoking + Drinking Detection | Detects smoking and drinking-related objects/actions | YOLO-based object detection |

---

## Project Architecture

```text
Input Source
Live Webcam / Recorded Driver Video
        |
        v
Video Preprocessing
- Frame extraction
- Face / upper body crop extraction
- Resize and normalize frames
- Fixed-length sequence creation
        |
        v
Multi-Module Driver Analysis
        |
        +-----------------------------+
        |                             |
        v                             v
Video Temporal Branch          Landmark Temporal Branch
CNN-LSTM / CNN-GRU / TCN       EAR, MAR, Head Pose, Gaze
        |                             |
        +-------------+---------------+
                      |
                      v
              Driver State Classifier
          Safe / Caution / High Risk
                      |
                      v
Additional Safety Modules
- Seat belt detection
- Phone usage detection
- Smoking detection
- Drinking detection
- Driver activity classification
                      |
                      v
Driver Wellness Score
                      |
                      v
Alerts / Logs / Trip Report