# AI-Powered Driver Wellness and Safety Monitoring System — Milestone 2

## 1. Introduction

Milestone 2 focuses on identifying, verifying, cleaning, preprocessing, and organizing the datasets required for the **AI-Powered Driver Wellness and Safety Monitoring System**. In Milestone 1, the project direction was expanded from a narrow drowsiness detection system to a broader driver wellness and safety monitoring system. Based on that direction, Milestone 2 concentrates on making the datasets ready for model development in Milestone 3.

The system is divided into multiple driver monitoring modules. Each module requires a different type of dataset and preprocessing pipeline. Video-based fatigue detection needs continuous driver videos, landmark-based temporal analysis needs extracted face and head-pose features, driver distraction classification needs activity images, and object detection modules need annotated image-label pairs in YOLO format.

The main goal of this milestone is not model training yet. The goal is to prove that the selected datasets are available, understandable, properly cleaned, split without leakage, and structured in a format that can be directly used in Milestone 3.

---

## 2. Milestone 2 Objective

The objective of Milestone 2 is to prepare model-ready datasets for the selected project modules. The submission addresses the following requirements:

- Identify and verify dataset sources.
- Document dataset ownership, format, classes, and usage constraints.
- Understand dataset size, class distribution, and metadata.
- Perform EDA and quality checks.
- Remove or handle corrupted, duplicate, invalid, or irrelevant samples.
- Prepare train, validation, and test splits.
- Prevent data leakage between splits.
- Create processed folder structures.
- Document preprocessing steps for reproducibility.
- Prepare hosted processed dataset links where available.
- Confirm readiness for model experiments in Milestone 3.

---

## 3. System Modules Covered

The project uses multiple datasets because the driver wellness system has multiple responsibilities.

| Team Member | Module | Dataset / Source Type | Target Task |
|---|---|---|---|
| Shiwani | Landmark-Based Temporal Features | YawDD | EAR, MAR, head pose temporal sequence modeling |
| Kushagra | Video-Based Fatigue Detection | UTA-RLDD | Safe / Caution / High Risk fatigue classification |
| Shubham | Driver Activity Classification | AUC Distracted Driver Dataset | Driver distraction image classification |
| Sohini | Seat Belt and Phone Usage Detection | DMS Dataset | YOLO object detection |
| Ravina | Smoking and Drinking Detection | YOLO-format smoking/drinking dataset | Object detection |

Each dataset was processed according to the model type planned for that module. Image classification datasets were organized into class-wise folders. Object detection datasets were converted into YOLO-compatible structures. Temporal feature datasets were stored as sequence arrays and metadata files. Video fatigue data was prepared as class-wise frame samples with a plan to extend it into frame sequences in Milestone 3.

---

## 4. Dataset Selection Summary

### 4.1 Landmark-Based Temporal Features

The landmark feature module uses the **YawDD dataset**. This dataset contains yawning and driver face videos and supports extraction of facial landmarks, eye features, mouth features, and head-pose signals. These signals are useful for temporal fatigue analysis because they can represent behavioral changes across time.

The extracted features include:

- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head pitch
- Head yaw
- Head roll

The final processed feature format is a temporal window of 30 frames with 5 features per frame.

### 4.2 Video-Based Fatigue Detection

The video fatigue detection module uses the **UTA Real-Life Drowsiness Dataset / UTA-RLDD**. For Milestone 2, the processed subset includes `Fold1_part1` and `Fold1_part2`, covering 12 subjects and 36 videos.

The original UTA-RLDD labels were mapped to the project classes:

| Original Label | Dataset Meaning | Project Class |
|---|---|---|
| `0` | Alert / Awake | Safe |
| `5` | Low Vigilance | Caution |
| `10` | Drowsy | High Risk |

This dataset was selected because fatigue is a temporal condition. A single frame may not clearly show whether the driver is tired, blinking normally, talking, or yawning. Therefore, video data is more appropriate for learning behavior over time.

### 4.3 Driver Activity Classification

The driver activity module uses the **AUC Distracted Driver Dataset**. This dataset supports image-based classification of driver activities.

The final selected classes are:

- other_activities
- safe_driving
- talking_phone
- texting_phone
- turning

This dataset supports a CNN-based activity classifier using models such as ResNet50, MobileNetV3, and EfficientNet-B0.

### 4.4 Seat Belt and Phone Usage Detection

The seat belt and phone usage module uses a filtered **DMS Driver Monitoring System** dataset. The dataset was processed into a clean two-class YOLO object detection dataset.

The final classes are:

| Class ID | Class Name |
|---:|---|
| 0 | Phone |
| 1 | Seatbelt |

This module is intended to detect visible safety-related objects and compliance signals from driver cabin images.

### 4.5 Smoking and Drinking Detection

The smoking and drinking module uses a YOLO-format object detection dataset organized into images, labels, and a `data.yaml` file. The processed dataset follows the standard YOLO train, validation, and test structure.

This module supports detection of unsafe driving behaviors involving smoking and drinking objects or actions.

---

## 5. Common Dataset Inspection and EDA

Across all modules, the team followed a common dataset inspection process before preprocessing.

The inspection process included:

1. Verifying folder structure.
2. Verifying class names and labels.
3. Counting samples in each class.
4. Checking for missing files.
5. Checking whether images or videos could be loaded.
6. Generating class distribution charts.
7. Visualizing sample images or frames.
8. Reviewing resolution, FPS, duration, brightness, or feature distributions depending on the dataset.
9. Identifying duplicates, corrupt files, invalid annotations, or inconsistent labels.
10. Confirming that each dataset could be split into train, validation, and test sets.

This common inspection step helped ensure that all datasets were suitable for the planned model pipelines.

---

## 6. Data Quality Assessment

### 6.1 Landmark-Based Temporal Features

For the landmark module, MediaPipe FaceLandmarker was used to extract features from the YawDD dataset. The pipeline verified landmark detection across 349 videos and 288,174 frames. Landmark extraction achieved a 99.35% success rate. Frames where landmarks could not be extracted were removed, resulting in 1,885 invalid frames being discarded and 286,289 frames being retained.

Additional checks included:

- head-pose stability validation
- EAR and MAR range validation
- landmark jitter review
- manual inspection of representative success and failure frames

The final output was considered reliable enough for temporal sequence modeling.

### 6.2 Video-Based Fatigue Detection

For the UTA-RLDD fatigue dataset, the downloaded subset contained 12 subjects and 36 videos. Automated scripts verified that all expected videos were present and readable.

| Check | Result |
|---|---:|
| Subjects processed | 12 |
| Videos found | 36 |
| Missing videos | 0 |
| Corrupt / unreadable videos | 0 |
| Safe videos | 12 |
| Caution videos | 12 |
| High Risk videos | 12 |

The dataset contained both `.mov` and `.mp4` files, so the preprocessing scripts were updated to support both formats. Video metadata such as FPS, duration, frame count, resolution, and file size was extracted using OpenCV.

Manual visual checks showed that individual frames from different fatigue classes can look similar. This confirms that temporal modeling is needed instead of relying only on isolated frames.

### 6.3 Driver Activity Classification

For the AUC distracted driver dataset, duplicate, corrupted, and blurry image checks were performed.

| Quality Issue | Count | Action |
|---|---:|---|
| Duplicate images | 2,969 | Removed |
| Corrupted images | 10 | Removed |
| Blurry images | 31 | Manually inspected and retained |

Duplicate detection was performed using perceptual hashing. Corrupted images were detected using OpenCV loading checks. Blurry images were identified using Laplacian variance. The blurry samples were retained because they represented realistic driving conditions and the count was small.

After cleaning, the dataset contained 4,307 clean images across the selected driver activity classes.

### 6.4 Seat Belt and Phone Detection

For the DMS seat belt and phone detection dataset, the team verified image-label parity and YOLO annotation validity.

Key checks included:

- every image had a corresponding annotation file
- YOLO bounding boxes followed the expected format
- normalized coordinate values stayed within valid bounds
- irrelevant classes were removed
- remaining classes were remapped to Phone and Seatbelt
- unannotated background frames were filtered out

The final dataset contained 6,140 images and 6,140 label files. A total of 6,701 bounding boxes were validated.

### 6.5 Smoking and Drinking Detection

The smoking and drinking dataset was organized in YOLO format with train, validation, and test folders. It includes images, labels, and a `data.yaml` file. The dataset was prepared using class harmonization, deduplication, augmentation, and split verification. The processed structure is compatible with YOLO-based training in Milestone 3.

---

## 7. Preprocessing Summary

Different preprocessing methods were used depending on the dataset type.

### 7.1 Common Preprocessing Steps

Across modules, the following preprocessing steps were performed:

- folder and label verification
- missing file checks
- corrupt file checks
- duplicate detection where applicable
- class distribution analysis
- sample visualization
- resizing
- normalization
- augmentation where applicable
- train/validation/test split creation
- processed folder or feature file generation

### 7.2 Image Classification Preprocessing

The driver activity images were cleaned, resized, normalized, and augmented. Images were resized to 224 × 224 pixels to support CNN-based models such as ResNet50, MobileNetV3, and EfficientNet-B0. Pixel values will be normalized during model training.

Augmentations included:

- horizontal flip
- random brightness and contrast
- random rotation

### 7.3 Video Fatigue Preprocessing

The fatigue detection videos were scanned and converted into a lightweight processed sample dataset. Five frames were extracted from each video at fixed timeline positions:

```text
10%, 30%, 50%, 70%, 90%
```

The final processed sample contains 180 frames. In Milestone 3, these frames will be extended into fixed-length sequences such as 16-frame, 32-frame, or 64-frame clips.

### 7.4 Landmark Feature Preprocessing

The landmark module extracted EAR, MAR, pitch, yaw, and roll from each frame. Invalid frames were removed. Features were normalized using statistics computed only from the training split. Fixed-length temporal windows of 30 frames with stride 15 were then generated.

### 7.5 YOLO Object Detection Preprocessing

For the object detection modules, datasets were organized into YOLO-compatible folder structures. This includes train, validation, and test folders containing images and labels. Each dataset also includes a `data.yaml` file.

For the seat belt and phone detection dataset, images were resized to 640 × 640 pixels, annotations were validated, and classes were remapped to a two-class schema.

---

## 8. Train / Validation / Test Split Strategy

The team used different split strategies depending on dataset type and available metadata.

| Feature | Split Strategy |
|---|---|
| Landmark Features | Subject-wise split |
| Video Fatigue Detection | Subject-wise split |
| Driver Activity Classification | 70% / 15% / 15% |
| Seat Belt & Phone Detection | 70% / 20% / 10% |
| Smoking & Drinking Detection | 80% / 10% / 10% |

Subject-wise splitting was used wherever subject IDs or video identity could cause leakage. For video and temporal modules, the same subject or video sequence must not appear in multiple splits. For image and object detection datasets, duplicate removal, class-wise splitting, original split boundary preservation, and stratification were used to reduce leakage risk.

---

## 9. Leakage Prevention

Leakage prevention was treated as a major Milestone 2 requirement because several datasets contain similar frames, video sequences, or repeated subjects.

### 9.1 Landmark Module

For the landmark module, all videos from one subject were assigned to only one split. All 30-frame temporal windows remained within their native video’s assigned split. Z-score normalization statistics were computed only from the training set and then applied to validation and test sets.

### 9.2 Fatigue Detection Module

For the fatigue module, a strict subject-level split was used.

| Split | Subject IDs | Videos | Processed Frames |
|---|---|---:|---:|
| Train | 01–08 | 24 | 120 |
| Validation | 09–10 | 6 | 30 |
| Test | 11–12 | 6 | 30 |

All three videos from a subject stayed in the same split. Frames were extracted only after the split was finalized. This prevents frames from the same video or same subject from appearing in multiple splits.

### 9.3 Driver Activity Module

For the distracted driver dataset, duplicate images were removed before splitting. A post-split hash check found possible near-duplicate images from consecutive sequences, so the team recommends using sequence-wise or subject-wise splitting if driver IDs become available later.

### 9.4 Seat Belt and Phone Module

For the seat belt and phone dataset, original split boundaries were preserved. Random frame-level migration was avoided because consecutive driving frames can contain nearly identical backgrounds. This helps prevent the model from memorizing scene-specific information.

### 9.5 Smoking and Drinking Module

For the smoking and drinking detection dataset, deduplication was performed before splitting, and augmentation was applied only to the training set. This avoids augmented versions of the same image appearing in validation or test data.

---

## 10. Processed Dataset Structure

Each module has a processed dataset structure based on its model requirements.

### 10.1 Landmark-Based Temporal Features

```text
processed_dataset/
├── raw_features_all_videos.csv
├── normalized_features_all_videos.csv
├── train_windows.npy
├── val_windows.npy
├── test_windows.npy
├── train_meta.csv
├── val_meta.csv
└── test_meta.csv
```

### 10.2 Video-Based Fatigue Detection

```text
data/
└── processed/
    └── fatigue_detection/
        ├── train/
        │   ├── Safe/
        │   ├── Caution/
        │   └── High_Risk/
        ├── val/
        │   ├── Safe/
        │   ├── Caution/
        │   └── High_Risk/
        └── test/
            ├── Safe/
            ├── Caution/
            └── High_Risk/
```

### 10.3 Driver Activity Classification

```text
processed_dataset/
├── train/
│   ├── other_activities/
│   ├── safe_driving/
│   ├── talking_phone/
│   ├── texting_phone/
│   └── turning/
├── validation/
│   └── class folders
└── test/
    └── class folders
```

### 10.4 Seat Belt and Phone Detection

```text
dms_processed_yolo/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 10.5 Smoking and Drinking Detection

```text
smoking_drinking_yolo/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## 11. Final Processed Dataset Summary

| Team Member | Feature Type | Final Format | Train / Validation / Test Split | Leakage Prevention |
|---|---|---|---|---|
| Shiwani | Landmark temporal features | `.npy` + `.csv` | Subject-wise | Subject-wise split, no shared windows, training-only normalization |
| Kushagra | Video fatigue detection | Class-wise image folders | Subject-wise | Subject-level split before frame extraction |
| Shubham | Driver activity classification | Image folders | 70% / 15% / 15% | Duplicate removal before split |
| Sohini | Seat belt and phone detection | YOLO images + labels + `data.yaml` | 70% / 20% / 10% | Sequence-based stratified split |
| Ravina | Smoking and drinking detection | YOLO images + labels + `data.yaml` | 80% / 10% / 10% | Deduplication before split and augmentation only on training set |

---

## 12. Hosted Dataset Links

The processed datasets are hosted or prepared for hosting through Google Drive.

| Team Member | Feature | Hosted Dataset Status |
|---|---|---|
| Shiwani | Landmark Features | [https://drive.google.com/drive/folders/1RdFLeJ66Qq\_8-60qwTNTnBSuhWUYacYZ?usp=sharing](https://drive.google.com/drive/folders/1RdFLeJ66Qq_8-60qwTNTnBSuhWUYacYZ?usp=sharing) |
| Kushagra | Video Fatigue Detection | [https://drive.google.com/file/d/1n\_Zf8Mt9JuxlZhLxPoo6VUvXK8WX-FV1/view?usp=sharing](https://drive.google.com/file/d/1n_Zf8Mt9JuxlZhLxPoo6VUvXK8WX-FV1/view?usp=sharing) |
| Shubham | Driver Activity Classification | [https://drive.google.com/drive/folders/1y92pQ918sTmLsfZALlRlJp84m1nYh1zO?usp=sharing](https://drive.google.com/drive/folders/1y92pQ918sTmLsfZALlRlJp84m1nYh1zO?usp=sharing) |
| Sohini | Seat Belt and Phone Detection | [https://drive.google.com/file/d/1PXem7gp6sEWVlX7Vwn46NFPm5j1-G2y7/view?usp=sharing](https://drive.google.com/file/d/1PXem7gp6sEWVlX7Vwn46NFPm5j1-G2y7/view?usp=sharing) |
| Ravina | Smoking and Drinking Detection | [https://drive.google.com/drive/folders/1v0PFbBx-Et2QJINfn0sghnSkBygL2OiD](https://drive.google.com/drive/folders/1v0PFbBx-Et2QJINfn0sghnSkBygL2OiD) |

Where a hosted link is not yet available, the dataset can be regenerated using the preprocessing scripts and documentation included with the project.

---

## 13. Model Input Format Summary

| Feature | Planned Model | Input Shape | Format | Status |
|---|---|---|---|---|
| Driver Activity Classification | ResNet50, MobileNetV3, EfficientNet-B0 | `(batch, 3, 224, 224)` | RGB image | Ready |
| Video Fatigue Detection | CNN-LSTM, CNN-GRU, TCN, lightweight 3D CNN | `(16, 224, 224, 3)` | Video sequence | Ready for sequence creation |
| Landmark Features | LSTM | `(30, 5)` | Temporal feature sequence | Ready |
| Seat Belt / Phone Detection | YOLOv8n, YOLO11n | `(640, 640, 3)` | RGB image + YOLO labels | Ready |
| Smoking / Drinking Detection | YOLO-based model | `(640, 640, 3)` | RGB image + YOLO labels | Ready |

The datasets are aligned with their intended Milestone 3 models. Classification datasets are ready for CNN training, temporal features are ready for LSTM training, and object detection datasets are ready for YOLO training.

---

## 14. Reproducibility

The team documented preprocessing steps and generated scripts or notebooks for dataset preparation. The reproducibility plan is:

1. Download the raw dataset from the documented source.
2. Run the dataset inventory script or notebook.
3. Validate missing files, corrupt files, and annotation format.
4. Generate metadata and EDA charts.
5. Apply cleaning steps such as duplicate removal or invalid frame removal.
6. Create train, validation, and test splits.
7. Save processed files into the final folder structure.
8. Use the hosted processed dataset where available.

This ensures that the processed datasets can be recreated if required.

---

## 15. Dataset Adequacy and Limitations

The selected datasets are adequate for Milestone 3 because each planned model has a corresponding processed dataset.

However, some limitations remain:

- Some datasets are based on controlled or limited environments.
- Real-world night, rain, glare, and occlusion cases may still be underrepresented.
- Video fatigue detection currently uses a subset of UTA-RLDD because the full dataset is large.
- Driver activity data may contain near-duplicate images from consecutive frames.
- Some datasets do not provide subject IDs, which limits perfect leakage prevention.
- Hosted links should be verified before final submission.
- Smoking and drinking detection details should be expanded further if additional dataset statistics become available.

These limitations will be considered during model training and evaluation in Milestone 3.

---

## 16. Connection to Milestone 3

Milestone 3 will focus on training and comparing models using the processed datasets prepared in this milestone.

Planned next steps include:

- Train image classification models for driver distraction.
- Train or fine-tune YOLO models for seat belt, phone, smoking, and drinking detection.
- Train LSTM-based models on landmark temporal features.
- Convert fatigue frames into fixed-length sequences and test CNN-LSTM, CNN-GRU, TCN, or lightweight 3D CNN models.
- Compare model accuracy, precision, recall, F1-score, mAP, FPS, and false alarm behavior.
- Integrate module outputs into a driver wellness score.

The current processed datasets are organized to directly support these next steps.

---

## 17. Team Member Contribution Summary

| Team Member | Milestone 2 Contribution |
|---|---|
| Kushagra | Prepared UTA-RLDD fatigue detection dataset, EDA, split strategy, processed sample frames, report section, and GitHub organization. |
| Shiwani | Prepared landmark temporal feature dataset, processed sequence files, and processed dataset hosting summary. |
| Shubham | Prepared driver activity preprocessing, model-readiness summary, input format details, and related dataset preparation. |
| Sohini | Prepared seat belt and phone detection dataset processing, data quality checks, leakage prevention summary, and YOLO structure. |
| Ravina | Prepared smoking and drinking detection dataset structure, hosting details, presentation/work-log support, and final review items. |

---

## 18. Team Review and Initials

Instead of handwritten signatures, initials are used because the repository is public.

| Sr. No. | Team Member Name | Responsibility | Review Initials |
|---:|---|---|---|
| 1 | Kushagra | Video fatigue dataset, final report assembly, GitHub structure | KB |
| 2 | Shiwani | Landmark temporal features, processed dataset hosting summary | ST |
| 3 | Shubham | Driver activity classification, preprocessing/model-readiness summary | ST |
| 4 | Sohini | Seat belt and phone detection, quality/leakage summary | SS |
| 5 | Ravina | Smoking and drinking detection, presentation and final review | R |

---

## 19. Conclusion

Milestone 2 establishes the dataset foundation for the AI-Powered Driver Wellness and Safety Monitoring System. The team identified datasets for all major modules, performed EDA and quality checks, created train/validation/test splits, prevented leakage where possible, and organized the processed outputs into model-ready structures.

The datasets are now prepared for Milestone 3 model development. The next milestone will focus on training, comparing, and evaluating the planned models for driver fatigue, landmark-based temporal signals, driver distraction, seat belt and phone usage, and smoking/drinking detection.
