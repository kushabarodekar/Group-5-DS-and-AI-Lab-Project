# **Milestone 3 Report**

## **AI-Powered Driver Wellness & Safety Monitoring System**

### **Model Architecture Design**

Team Members: Kushagra, Shiwani, Shubham, Sohini, Ravina

## **1\. Introduction**

### **1.1 Overview**

The objective of Milestone 3 is to finalize the deep learning architectures for all modules of the AI-Powered Driver Wellness & Safety Monitoring System. Unlike Milestone 2, which focused on dataset preparation and preprocessing, this milestone defines the complete model design that will be implemented and trained in Milestone 4\.

The proposed system consists of five independent deep learning modules that work together to monitor the driver's behavior and estimate an overall driver wellness score. Each module is responsible for detecting a specific safety-related event. The outputs from all modules are fused by a centralized risk assessment engine to generate a comprehensive driver safety report suitable for fleet management platforms such as Uber, Ola, and Rapido.

### **1.2 Objectives**

The objectives of Milestone 3 are:

- Select appropriate deep learning architectures for each module  
    
- Justify the selection of each model  
    
- Define the input and output specifications  
    
- Select suitable loss functions and evaluation metrics  
    
- Plan the hyperparameters for future training  
    
- Estimate computational requirements  
    
- Design the end-to-end inference pipeline  
    
- Prepare the system for implementation in Milestone 4

## **2\. Overall System Architecture**

The proposed Driver Wellness and Safety Monitoring System receives video streams captured from an in-vehicle camera. Different deep learning models analyze various aspects of driver behavior in parallel. Each model generates an independent prediction, which is forwarded to a centralized Risk Fusion Engine. The fusion engine combines all predictions to estimate the driver's overall wellness score and generate a detailed safety report.

                 Driver Camera

                      │

                      ▼

              Video Stream Input

                      │

  ┌───────────────────┼────────────────────┐

  │                   │                    │

  ▼                   ▼                    ▼

Driver Activity      Seatbelt & Phone      Smoking/Drinking

MobileNetV3            YOLOv8n               YOLOv8n

  │                   │                    │

  └──────────────┬────┴─────────────┐

                 │                  │

                 ▼                  ▼

      Video Fatigue         Landmark Fatigue

       CNN-LSTM                  LSTM

                 │

                 ▼

          Risk Fusion Engine

                 │

                 ▼

      Driver Wellness Score

                 │

                 ▼

  Driver Report / Uber Dashboard

**FIGURE 1: End-to-End System Architecture Diagram**

## **3\. Module 1 – Video-Based Fatigue Detection**

### **3.1 Objective**

Detect fatigue levels from temporal driver video sequences.

### **3.2 Candidate Models**

- CNN-LSTM  
    
- CNN-GRU  
    
- Temporal Convolutional Network (TCN)  
    
- Lightweight 3D CNN

### **3.3 Final Model Selection**

**Selected Model:** CNN-LSTM

**Justification**

Three temporal architectures (CNN-GRU, CNN-LSTM, and Tuned CNN-LSTM) were considered for video-based fatigue detection. CNN-LSTM was selected because it provides better long-term temporal feature learning while maintaining good computational efficiency. It achieved the best overall performance among the evaluated architectures and is suitable for modeling fatigue progression across consecutive video frames.

### **3.4 Input Specification**

| Parameter | Value |
| :---- | :---- |
| Input Type | Video Sequence |
| Input Shape | (16, 224, 224, 3\) |
| Output Classes | Safe, Caution, High Risk |

### **3.5 Loss Function**

CrossEntropyLoss \- Suitable for multi-class classification.

### **3.6 Evaluation Metrics**

Accuracy, Precision, Recall, F1-score, Confusion Matrix

### **3.7 Planned Hyperparameters**

| Hyperparameter | Planned Value |
| :---- | :---- |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 8 |
| Epochs | 30 |

### **3.8 Computational Requirements**

- Framework: PyTorch  
    
- GPU: 16 GB recommended  
    
- RAM: 16 GB  
    
- Hardware: Apple M4 Pro / NVIDIA Tesla T4

### **3.9 Architecture Diagram**

Selected Model Architecture

                Input Video

                     │

                     ▼

         Frame Sampling (5 FPS)

                     │

                     ▼

        Sequence Generation (16 Frames)

                     │

                     ▼

 ┌───────────────────────────────────┐

 │         CNN Feature Encoder       │

 │                                   │

 │   Conv2D (3 → 32\\)                 │

 │   ReLU                            │

 │   MaxPool                         │

 │                                   │

 │   Conv2D (32 → 64\\)                │

 │   ReLU                            │

 │   MaxPool                         │

 │                                   │

 │   Conv2D (64 → 128\\)               │

 │   ReLU                            │

 │   MaxPool                         │

 │                                   │

 │   Conv2D (128 → 256\\)              │

 │   ReLU                            │

 │   AdaptiveAvgPool (1×1)           │

 └───────────────────────────────────┘

                     │

      256-D Feature Vector per Frame

                     │

                     ▼

          LSTM (Hidden Size \\= 128\\)

                     │

            Final Hidden State

                     │

                     ▼

             Dropout (0.30)

                     │

                     ▼

     Fully Connected Layer (128 → 3\\)

                     │

                     ▼

          Softmax Classification

                     │

 ┌─────────────┬─────────────┬─────────────┐

 │    Safe     │   Caution   │  High Risk  │

 └─────────────┴─────────────┴─────────────┘

**FIGURE 2: Video-Based Fatigue Detection Architecture**

The proposed pipeline extracts spatial features from each video frame using a CNN encoder. The extracted features are then passed to an LSTM network to model temporal dependencies across frame sequences before classifying the driver's fatigue level.

## **4\. Module 2 – Landmark-Based Fatigue Detection**

### **4.1 Objective**

Detect fatigue using facial landmark sequences.

### **4.2 Candidate Models**

- LSTM  
    
- GRU  
    
- TCN  
    
- MLP

### **4.3 Final Model Selection**

**Selected Model:** LSTM

**Justification**

LSTM was selected because fatigue-related facial features such as EAR, MAR, head pitch, yaw, and roll evolve over time. Compared with MLP, GRU, and TCN, LSTM is better suited for learning long-term temporal dependencies in sequential facial landmark data while maintaining stable performance.

### **4.4 Input Specification**

| Parameter | Value |
| :---- | :---- |
| Features | EAR, MAR, Pitch, Yaw, Roll |
| Input Shape | (30, 5\) |
| Output Classes | Alert, Mild Fatigue, Drowsy |

### **4.5 Loss Function**

CrossEntropyLoss

### **4.6 Evaluation Metrics**

Accuracy, Precision, Recall, F1-score

### **4.7 Planned Hyperparameters**

- Learning Rate: 0.001  
    
- Batch Size: 32  
    
- Epochs: 30

### **4.8 Feature Extraction Pipeline**

The model receives sequences of facial features extracted using MediaPipe Face Landmarker. Each sequence contains 30 consecutive frames with five features (EAR, MAR, Pitch, Yaw, Roll), which are normalized before being passed to the LSTM network for fatigue classification.

### **4.8 Architecture Diagram**

YawDD Videos

    │

    ▼

MediaPipe Face Landmarker

    │

    ▼

Facial Landmark Extraction

    │

    ▼

EAR

MAR

Pitch

Yaw

Roll

    │

    ▼

Feature Cleaning

    │

    ▼

Normalization

    │

    ▼

Sliding Window

(30 Frames)

    │

    ▼

Train / Validation / Test Split

    │

    ▼

MLP

LSTM

GRU

TCN

    │

    ▼

Hyperparameter Tuning

    │

    ▼

Best Model (LSTM)

    │

    ▼

Prediction

    │

    ▼

Normal

Talking

Yawning

Talking\\\_Yawning

**FIGURE 3: Landmark-Based Fatigue Detection Architecture**

## **5\. Module 3 – Driver Activity Classification**

### **5.1 Objective**

Classify the driver's activity from RGB images to identify distracting behaviors.

### **5.2 Candidate Models**

- MobileNetV3  
    
- ResNet50  
    
- EfficientNet-B0

### **5.3 Model Comparison**

| Model | Advantages | Limitations |
| :---- | :---- | :---- |
| MobileNetV3 | Lightweight, fast inference, low memory, real-time capable | Slightly lower accuracy than larger networks |
| ResNet50 | Excellent feature extraction, high accuracy | Large model size, slower inference, high compute |
| EfficientNet-B0 | Good accuracy-efficiency balance | More complex than MobileNetV3, slower on edge |

### **5.4 Final Model Selection**

Selected Model: **MobileNetV3**

***Justification***:

Although **EfficientNet-B0** achieved the highest classification accuracy on the candidate dataset and **ResNet50** also demonstrated strong performance, **MobileNetV3** was selected as the final architecture due to its significantly lower computational complexity, faster inference speed, and lower memory consumption.

The proposed Driver Wellness & Safety Monitoring System is intended for real-time deployment on resource-constrained platforms such as Raspberry Pi, NVIDIA Jetson Nano, and other in-vehicle embedded systems. In such applications, low inference latency, reduced power consumption, and efficient memory utilization are more important than a small improvement in classification accuracy.

MobileNetV3 is specifically designed for efficient edge deployment through lightweight convolutional blocks, depthwise separable convolutions, squeeze-and-excitation modules, and neural architecture search (NAS)-based optimization. These characteristics enable real-time inference while maintaining competitive classification performance.

Therefore, considering both the experimental results and the deployment requirements of the proposed system, **MobileNetV3 provides the best trade-off between classification performance and computational efficiency**, making it the most suitable architecture for this project.

**5.5 Architecture Justification**

| Requirement | MobileNetV3 | Justification |
| :---- | :---- | :---- |
| Real-time inference | **✓** | Optimized for fast image classification with low inference latency. |
| Low latency | ✓ | Enables real-time processing of continuous driver camera frames. |
| Lightweight architecture | **✓** | Uses significantly fewer parameters than ResNet50 and EfficientNet-B0. |
| Suitable for edge devices | **✓** | Specifically designed for deployment on embedded and mobile platforms. |
| Low memory usage | **✓** | Requires less GPU memory, making it suitable for resource-constrained systems. |
| Good classification accuracy | **✓** | Provides competitive accuracy while maintaining high computational efficiency. |
| Embedded deployment | **✓** | Can be deployed on Raspberry Pi, NVIDIA Jetson Nano, and similar edge AI devices. |
| Energy efficient | **✓** | Lower computational complexity results in reduced power consumption, making it appropriate for in-vehicle systems. |
| Scalability | **✓** | Can be integrated into larger driver monitoring systems with minimal computational overhead. |

### **5.5.1 MobileNetV3 Architectural Overview**

MobileNetV3 is a lightweight convolutional neural network designed specifically for real-time image classification on resource-constrained edge devices. Unlike conventional deep neural networks such as ResNet50, MobileNetV3 reduces computational complexity while maintaining competitive classification accuracy through several architectural innovations. These include **depthwise separable convolutions**, **inverted residual blocks**, **linear bottlenecks**, **Squeeze-and-Excitation (SE) modules**, and the **h-swish activation function**. Together, these components significantly reduce the number of trainable parameters, floating-point operations (FLOPs), memory consumption, and inference latency, making MobileNetV3 suitable for embedded driver monitoring systems.

### **Standard Convolution**

In a standard convolution, every convolutional filter operates across all input channels simultaneously to generate each output feature map. Although this approach extracts rich spatial features, it requires a large number of parameters and computational operations, resulting in higher memory usage and slower inference.

### **Depthwise Separable Convolution**

MobileNetV3 replaces standard convolution with **depthwise separable convolution**, which consists of two consecutive operations:

1. **Depthwise Convolution:** Applies one filter independently to each input channel to extract spatial features.  
2. **Pointwise Convolution (1×1):** Combines information from all channels to generate new feature representations.

By separating spatial filtering from channel mixing, depthwise separable convolution drastically reduces computational cost and the number of parameters while preserving most of the model's predictive performance.

### **Inverted Residual Block**

Instead of compressing features as in traditional residual networks, MobileNetV3 first expands the feature dimensions, performs lightweight depthwise convolution, and then compresses them back to a lower-dimensional representation. A shortcut (skip) connection is used whenever the input and output dimensions are identical, allowing information to flow directly through the network. This design improves feature reuse, accelerates training, and enables deeper networks with minimal computational overhead.

### **Linear Bottleneck**

The final projection layer of each inverted residual block uses a **linear activation** instead of a nonlinear activation function. This preserves important low-dimensional feature information that could otherwise be lost due to nonlinear transformations, thereby improving feature representation while keeping the model compact.

### **Squeeze-and-Excitation (SE) Module**

The SE module introduces a lightweight channel attention mechanism. It first summarizes global information from each feature map using global average pooling (squeeze) and then learns channel-wise importance weights (excitation). More informative channels are emphasized while less useful channels are suppressed, improving feature quality with only a small increase in computational cost.

### **h-swish Activation Function**

MobileNetV3 employs the **h-swish activation function**, an efficient approximation of the swish activation. Compared with the traditional ReLU activation, h-swish improves gradient flow and feature learning while requiring very little additional computation. This contributes to higher classification accuracy without sacrificing inference speed.

### **Overall Architectural Advantage**

Compared with ResNet50 and EfficientNet-B0, MobileNetV3 provides the best balance between computational efficiency and classification performance. In our experiments, MobileNetV3 required only **5.48 million parameters** and **231.32 MMac** while achieving competitive classification performance with an inference latency of approximately **22.7 ms per image**. In contrast, ResNet50 required **25.56 million parameters** and **4.13 GMac**, making it significantly more computationally expensive, while EfficientNet-B0 required **410.2 MMac** despite having similar latency. Although EfficientNet-B0 achieved the highest classification accuracy, MobileNetV3 was selected because its substantially lower computational complexity and memory requirements make it more suitable for real-time deployment on embedded platforms such as Raspberry Pi and NVIDIA Jetson Nano.

### **5.6 Candidate Dataset**

A representative subset (20% of full dataset, 4,200 images) was created using stratified random sampling across all five classes, preserving the original class distribution for rapid experimentation.

### **5.7 Input Specification**

| Parameter | Value |
| :---- | :---- |
| Input Type | RGB Image |
| Image Size | 224 × 224 × 3 |
| Normalization | ImageNet mean/std |

### **5.8 Output Classes**

- Safe Driving  
    
- Texting on Phone  
    
- Talking on Phone  
    
- Turning  
    
- Other Activities

### **5.9 Loss Function**

CrossEntropyLoss \- Standard for multi-class classification.

### **5.10 Baseline Performance**

| Model(Epoch) | Accuracy | Precision | Recall | F1-Score |
| :---- | :---- | :---- | :---- | :---- |
| MobileNetV3(15) | 89.35% | 89.74% | 89.35% | 89.39% |
| ResNet50(10) | 92.13% | 92.32% | 92.13% | 92.14% |
| EfficientNet-B0(10) | 93.98% | 94.08% | 93.98% | 93.98% |

*Baseline performance was obtained using ImageNet-pretrained weights on the representative candidate dataset after training for 10 epochs with the default hyperparameters (Adam optimizer, learning rate \= 0.001, batch size \= 32). These results serve as the reference for all subsequent hyperparameter tuning experiments.*

### **5.11 Hyperparameter Tuning Results (MobileNetV3)**

| Experiment | LR | Batch Size | Optimizer | Accuracy |
| :---- | :---- | :---- | :---- | :---- |
| Baseline | 0.001 | 32 | Adam | 87.04% |
| Exp 2 | 0.0005 | 32 | Adam | 90.74% |
| Exp 3 | 0.0001 | 32 | Adam | 87.96% |
| Exp 4 | 0.001 | 16 | Adam | 87.96% |
| Exp 5 | 0.001 | 64 | Adam | 68.06% |
| Exp 6 | 0.001 | 32 | SGD | 42.13% |
| Exp 7 | 0.001 | 32 | AdamW | 89.81% |

Optimal Configuration: LR=0.0005, Batch Size=32, Optimizer=Adam (90.74% accuracy).

The hyperparameter tuning experiments demonstrated that reducing the learning rate from 0.001 to 0.0005 improved the classification performance of MobileNetV3. Experiments with larger batch sizes resulted in unstable training for MobileNetV3, while the SGD optimizer showed considerably slower convergence compared with Adam. Based on these observations, Adam with a learning rate of 0.0005 and batch size of 32 was selected as the planned configuration for Milestone 4\.

### **5.12 Comparison Plots**

![FIGURE 4: Accuracy Comparison Across Experiments](Images/image5.png)


**FIGURE 4: Accuracy Comparison Across Experiments**

### **5.13 Training Pipeline**

Dataset → Train/Val Split (80/20) → Data Augmentation → Data Loader → MobileNetV3 → CrossEntropyLoss → Backpropagation (Adam, LR=0.0005) → Validation → Best Model Saved

### **5.14 Validation Strategy**

- Training: 80% (3,360 images)  
    
- Validation: 20% (840 images)  
    
- Stratified random split  
    
- Early stopping with patience \= 5 epochs

### **5.15 Evaluation Methodology**

Candidate Dataset

    │

    ▼

Train Dataset

    │

    ▼

Train MobileNetV3

    │

    ▼

Validation Dataset

    │

    ▼

Best Model Selection

    │

    ▼

Independent Test Dataset

    │

    ▼

Performance Evaluation

    │

    ├── Accuracy

    ├── Precision

    ├── Recall

    ├── F1-score

    └── Confusion Matrix

The candidate dataset was divided into training, validation, and independent testing subsets. The MobileNetV3 model was trained using the training dataset, while the validation dataset was used to monitor model performance and select the best-performing checkpoint. After training, the selected model was evaluated on an independent test dataset using Accuracy, Precision, Recall, F1-score, and the Confusion Matrix to obtain an unbiased estimate of its generalization performance.

### **5.16 Inference Pipeline**

RGB Driver Image

    │

    ▼

Resize (224×224)

    │

    ▼

Normalization

    │

    ▼

MobileNetV3 Backbone

    │

    ▼

Global Average Pooling

    │

    ▼

Fully Connected Layer

    │

    ▼

Softmax

    │

    ▼

Predicted Driver Activity

    │

    ▼

Risk Fusion Engine

### **5.17 Computational Requirements**

| Component | Specification |
| :---- | :---- |
| Framework | PyTorch |
| GPU | NVIDIA Tesla T4 / RTX 3060 or higher |
| GPU Memory | Minimum 4 GB (Recommended 6 GB+) |
| RAM | Minimum 8 GB (Recommended 16 GB) |
| Inference Speed | \~12.5 ms/image |
| CPU | Quad-Core Processor |
| Storage | 10 GB Free Space |

### **5.18 Architecture Diagram**

(224 × 224 × 3\)  
│  
▼  
Resize & Normalize  
│  
▼  
Initial Conv 3×3 Layer  
│  
▼  
MobileNetV3 Backbone  
(Inverted Residual Bottleneck Blocks  
\+ Depthwise Separable Convolutions  
\+ SE Modules \+ h-swish)  
│  
▼  
Global Average Pooling  
│  
▼  
Fully Connected Layer  
│  
▼  
Softmax Layer  
│  
▼  
Driver Activity Class  
(Safe, Texting, Talking, Turning, Other)

**FIGURE 5: MobileNetV3 Architecture Diagram**

### **References**

1. Howard et al., *Searching for MobileNetV3*, ICCV 2019\.  
     
2. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016\.  
     
3. Tan & Le, *EfficientNet*, ICML 2019\.  
     
4. Jocher et al., *YOLOv8 Documentation*, Ultralytics.  
     
5. Ultralytics, *YOLO11 Documentation*.  
     
6. Hochreiter & Schmidhuber, *Long Short-Term Memory*, Neural Computation, 1997\.  
     
7. Cho et al., *Learning Phrase Representations using RNN Encoder–Decoder (GRU)*, EMNLP 2014\.  
     
8. Bai et al., *Temporal Convolutional Networks*, 2018\.

## **6\. Module 4 – Seat Belt and Phone Detection**

### **6.1 Objective**

Detect seat belt usage and mobile phone usage.

### **6.2 Candidate Models**

- YOLOv8n  
    
- YOLO11n  
    
- YOLOv8s

### **6.3 Final Model Selection**

**Selected Model:** YOLOv8n

**Justification**

YOLOv8n was selected because it provides an excellent balance between detection accuracy, computational efficiency, and inference speed. Compared with YOLO11n and YOLOv8s, it is more suitable for real-time driver monitoring on edge devices such as NVIDIA Jetson and Raspberry Pi while maintaining reliable seat belt and phone detection performance.

### **6.4 Input Specification**

| Parameter | Value |
| :---- | :---- |
| Input Type | RGB Image |
| Image Size | 640 × 640 |

### **6.5 Output Classes**

- Phone  
    
- Seat Belt

### **6.6 Loss Function**

- Box Loss  
    
- Classification Loss  
    
- Distribution Focal Loss (DFL)

### **6.7 Evaluation Metrics**

mAP@50, mAP@50-95, Precision, Recall

### **6.8 Computational Requirements**

| Component | Specification |
| :---- | :---- |
| Framework | PyTorch |
| GPU | NVIDIA Tesla T4 / RTX 3060 or higher |
| GPU Memory | Minimum 4 GB (Recommended 6 GB+) |
| RAM | Minimum 8 GB (Recommended 16 GB) |
| Inference Speed | \~12.5 ms/image |
| CPU | Quad-Core Processor |
| Storage | 10 GB Free Space |

## **7\. Module 5 – Smoking and Drinking Detection**

### **7.1 Objective**

Detect smoking and drinking activities inside the vehicle.

### **7.2 Candidate Models**

- YOLOv8n  
    
- YOLO11n  
    
- YOLOv8s

### **7.3 Final Model Selection**

**Selected Model:** YOLOv8n

**Justification**

YOLOv8n was selected because it provides the best trade-off between detection accuracy, inference speed, and computational efficiency for real-time smoking and drinking detection. Its lightweight architecture makes it suitable for deployment in embedded dr*iver monitoring systems while maintaining reliable object localization.*

### **7.4 Input Specification**

| Parameter | Value |
| :---- | :---- |
| Input Type | RGB Image |
| Image Size | 640 × 640 |

### **7.5 Output Classes**

- Smoking  
    
- Drinking

### **7.6 Loss Function**

YOLO Detection Loss

### **7.7 Evaluation Metrics**

mAP@50, mAP@50-95, Precision, Recall

### **7.8 Architecture Diagram**

Figure 7: YOLOv8n architecture diagram (Smoking/Drinking Detection)                  Input Image  
(640 × 640\)  
│  
▼  
┌─────────────────┐  
│   Backbone      │  
│   (CSP Darknet) │  
│                 │  
│   Conv 3×3      │  
│   C2f Blocks    │  
│   SPPF Layer    │  
└─────────────────┘  
│  
▼  
┌─────────────────┐  
│   Neck          │  
│   (PAN-FPN)     │  
│                 │  
│   Multi-scale   │  
│   Feature Fusion│  
└─────────────────┘  
│  
▼  
┌─────────────────┐  
│  Detection Head │  
│                 │  
│  3 Detection    │  
│  Scales         │  
│  (P3, P4, P5)   │  
└─────────────────┘  
│  
▼  
┌─────────────────┐  
│  Output         │  
│                 │  
│  ○ Smoking      │  
│  ○ Drinking     │  
│  □ Bounding Box │  
│  % Confidence   │  
└─────────────────┘

**FIGURE 6: YOLOv8n Architecture for Smoking & Drinking Detection**

The YOLOv8n architecture consists of three main components: a CSP-based backbone for feature extraction, a PAN-FPN neck for multi-scale feature fusion, and an anchor-free detection head. The detection head predicts object classes (Smoking, Drinking) and bounding boxes at three different scales (P3, P4, P5) to detect objects of varying sizes. The architecture is optimized for real-time inference on edge devices with minimal computational overhead.

### **7.9 Architecture Summary**

The detector consists of a CSP-based backbone for feature extraction, a PAN-FPN neck for multi-scale feature fusion, and an anchor-free detection head using Distribution Focal Loss (DFL) for accurate object localization.

## **8\. Model Comparison Summary**

| Module | Candidate Models | Selected Model | Input Shape | Output |
| :---- | :---- | :---- | :---- | :---- |
| Video Fatigue | CNN-LSTM, CNN-GRU, TCN, 3D CNN | CNN-LSTM | (16,224,224,3) | 3 Classes |
| Landmark Fatigue | LSTM, GRU, TCN, MLP | LSTM | (30,5) | 3 Classes |
| Activity | MobileNetV3, ResNet50, EfficientNet-B0 | MobileNetV3 | (224,224,3) | 5 Classes |
| Seat Belt | YOLOv8n, YOLO11n, YOLOv8s | YOLOv8n | 640×640 | 2 Classes |
| Smoking | YOLOv8n, YOLO11n, YOLOv8s | YOLOv8n | 640×640 | 2 Classes |

## **9\. Computational Requirements**

| Module | Framework | GPU | Estimated Memory |
| :---- | :---- | :---- | :---- |
| Video Fatigue | PyTorch | NVIDIA T4 | 8–16 GB |
| Landmark Fatigue | PyTorch | NVIDIA T4 | 4–8 GB |
| Activity | PyTorch | NVIDIA T4 | 4–8 GB |
| Seat Belt | Ultralytics | NVIDIA T4 | 4–8 GB |
| Smoking | Ultralytics | NVIDIA T4 | 4–8 GB |

## **10\. Training Strategy for Milestone 4**

Each module will be trained using its respective processed dataset with the following common strategy:

- Optimizers: Adam/AdamW  
    
- Learning Rate: 0.0001 \- 0.001 (tuned per module)  
    
- Batch Size: 8-32 (based on module)  
    
- Epochs: 30-50  
    
- Augmentation: Random flip, rotation, brightness adjustment  
    
- Checkpointing: Best model saved based on validation performance  
    
- Early Stopping: Patience \= 5-10 epochs

## **11\. Expected Driver Wellness Pipeline**

The outputs from all five modules are combined as follows:

Camera Feed

  ↓

──────────────────────────────

Driver Activity Model   → Activity Class

Seatbelt Model          → Seatbelt Status

Smoking/Drinking Model  → Unsafe Behavior

Fatigue Video Model     → Fatigue Level

Landmark Fatigue Model  → Drowsiness Level

──────────────────────────────

  ↓

Risk Fusion Engine

  ↓

Driver Wellness Score

  ↓

Driver Report Generation

  ↓

Uber/Ola/Rapido Dashboard

### **Module Outputs and Fusion**

| Module | Model | Output | Risk Contribution |
| :---- | :---- | :---- | :---- |
| Driver Activity | MobileNetV3 | Activity Class (Safe/Distracted) | High (Immediate Safety Risk) |
| Seatbelt Detection | YOLOv8n | Seatbelt Status (On/Off) | High (Safety Compliance) |
| Smoking/Drinking | YOLOv8n | Unsafe Behavior Detected | Medium (Health/Safety Risk) |
| Video Fatigue | CNN-LSTM/TCN | Fatigue Level (Safe/Caution/High Risk) | High (Accident Risk) |
| Landmark Fatigue | LSTM/GRU | Drowsiness Level (Alert/Mild/Drowsy) | High (Accident Risk) |

### **Risk Fusion Engine**

The Risk Fusion Engine aggregates outputs from all five modules using a weighted scoring mechanism:

**Driver Wellness Score \= Σ (Module\_Weight × Module\_Risk\_Score)**

**Driver Wellness Score \= 0.25A \+ 0.15S \+ 0.10D \+ 0.25Fv ​+ 0.25Fl​**

where

* A \= Driver Activity Risk  
* S \= Seat Belt Risk  
* D \= Smoking Risk  
* Fv \= Video Fatigue  
* Fl \= Landmark Fatigue

### **Final Output**

The system generates a comprehensive Driver Wellness & Safety Report containing:

- Overall wellness score (0-100)  
    
- Individual module predictions  
    
- Risk level indicators  
    
- Recommendations for fleet managers  
    
- Historical trend analysis

Driver Camera

  │

  ▼

────────────────────────────

Driver Activity

Seatbelt Detection

Smoking Detection

Video Fatigue

Landmark Fatigue

────────────────────────────

  │

  ▼

Risk Fusion Engine

  │

  ▼

Driver Wellness Score

  │

  ▼

Safety Report

  │

  ▼

Uber / Ola / Rapido Dashboard

**FIGURE 7: End-to-End Driver Wellness Pipeline**

## **12\. Future Work**

In Milestone 4, the selected architectures for all five modules will be trained using their respective prepared datasets and optimized hyperparameters. Each module will undergo model training, validation, hyperparameter refinement, and performance evaluation. Finally, the trained models will be integrated through the Risk Fusion Engine to generate the overall Driver Wellness & Safety Report.

## **13\. Model Selection Rationale**

The selected architectures were chosen by considering accuracy, computational efficiency, inference speed, and suitability for real-time deployment. Lightweight deep learning models were prioritized to enable deployment on edge devices while maintaining reliable performance for driver monitoring tasks.

| Module | Final Model | Primary Reason |
| :---- | :---- | :---- |
| Video Fatigue | CNN-LSTM | Temporal sequence modeling |
| Landmark Fatigue | LSTM | Sequential facial feature analysis |
| Driver Activity | MobileNetV3 | Lightweight image classification |
| Seat Belt Detection | YOLOv8n | Fast object detection |
| Smoking Detection | YOLOv8n | Efficient real-time object detection |

## **14\. Conclusion**

Milestone 3 establishes the complete architectural design for the Driver Wellness and Safety Monitoring System. Suitable deep learning models have been selected for each module, along with their input/output specifications, loss functions, evaluation metrics, and computational requirements. The entire system has been designed to support modular development while enabling integration through a centralized Risk Fusion Engine. With the architecture finalized, the project is now fully prepared for model implementation and training in Milestone 4\.  
