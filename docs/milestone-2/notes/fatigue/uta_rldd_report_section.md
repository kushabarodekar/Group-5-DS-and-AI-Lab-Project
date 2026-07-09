# Video-Based Fatigue Detection Dataset: UTA-RLDD

## Dataset Selection

For the video-based fatigue detection module, we selected the **UTA Real-Life Drowsiness Dataset (UTA-RLDD)**. This dataset was selected because it contains real driver videos with different drowsiness levels and supports temporal fatigue detection.

The main reason for selecting this dataset is that fatigue is not always visible in a single image frame. A driver may appear normal in one frame, but their fatigue level can be understood more clearly by observing behavior across time. UTA-RLDD is suitable for this requirement because it provides video data that can be converted into frame sequences for models such as CNN-LSTM, CNN-GRU, TCN, or lightweight 3D CNN.

## Dataset Source and Ownership

| Field | Details |
|---|---|
| Dataset Name | UTA Real-Life Drowsiness Dataset / UTA-RLDD |
| Official Source | https://sites.google.com/view/utarldd/home |
| Download Source | Official Google Drive folder linked from the dataset page |
| Owner / Creator | University of Texas at Arlington / dataset authors |
| Dataset Type | RGB driver video dataset |
| Usage | Academic/research use, subject to dataset source terms |
| Module Supported | Video-based fatigue detection |

## Dataset Classes

The dataset contains three drowsiness levels for each subject.

| Video File | Dataset Class | Project Class |
|---|---|---|
| `0.mov` / `0.mp4` | Alert / Awake | Safe |
| `5.mov` / `5.mp4` | Low Vigilance | Caution |
| `10.mov` / `10.mp4` | Drowsy | High Risk |

The project-level class mapping is useful because our final system classifies driver condition into three levels:

- **Safe**
- **Caution**
- **High Risk**

## Downloaded Dataset Subset

The full UTA-RLDD dataset is large, approximately 96 GB. Due to local storage and processing constraints, the current Milestone 2 work was performed on the downloaded subset containing `Fold1_part1` and `Fold1_part2`.

| Fold | Subjects Included | Number of Subjects |
|---|---|---:|
| Fold1_part1 | 01, 02, 03, 04, 05, 06 | 6 |
| Fold1_part2 | 07, 08, 09, 10, 11, 12 | 6 |

Total subjects currently processed: **12**  
Total videos currently processed: **36**

Each subject contains one video for each class: Safe, Caution, and High Risk.

## Dataset Inventory Summary

The raw dataset was scanned using a Python inventory script. The script checked subject folders, video files, label codes, project class mapping, file type, and missing expected files.

| Check | Result |
|---|---:|
| Total subjects scanned | 12 |
| Total videos found | 36 |
| Missing expected files | 0 |
| Videos per subject | 3 |
| Classes | Safe, Caution, High Risk |

## Class Distribution

The currently downloaded subset is balanced.

| Project Class | Number of Videos |
|---|---:|
| Safe | 12 |
| Caution | 12 |
| High Risk | 12 |

This balance is useful because all three fatigue states are equally represented in the current subset.

## File Format Distribution

The dataset contains mixed video formats.

| File Format | Number of Videos |
|---|---:|
| `.mp4` | 21 |
| `.mov` | 15 |

Since both `.mov` and `.mp4` files are present, all dataset scanning and preprocessing scripts were updated to support both formats.

## Video Metadata Summary

The video metadata was extracted using OpenCV.

| Metadata Field | Observation |
|---|---|
| Total videos scanned | 36 |
| Videos opened successfully | 36 |
| Videos failed to open | 0 |
| Duration range | 7.16 to 11.55 minutes |
| Average duration | 9.98 minutes |
| FPS range | Approximately 15 to 30 FPS |
| Resolutions observed | Multiple portrait and landscape resolutions |

The videos have different resolutions and FPS values, so resizing and standard frame sampling are required before model training.

## Resolution Distribution

The dataset contains multiple resolutions, including portrait and landscape orientations. Examples observed include:

- 1080x1920
- 1280x720
- 720x1280
- 1920x1080
- 1440x2560
- 1080x720
- 848x480
- 480x848
- 480x720
- 320x568

This confirms the need for resizing during preprocessing.

## Exploratory Data Analysis

The following EDA outputs were generated:

| EDA Output | File |
|---|---|
| Class distribution chart | `01_class_distribution.png` |
| Dataset class distribution chart | `02_dataset_class_distribution.png` |
| File format distribution chart | `03_file_format_distribution.png` |
| Resolution distribution chart | `04_resolution_distribution.png` |
| FPS distribution chart | `05_fps_distribution.png` |
| Duration distribution chart | `06_duration_distribution.png` |
| Frame count distribution chart | `07_frame_count_distribution.png` |
| File size distribution chart | `08_file_size_distribution.png` |
| Sample frame grid | `09_sample_frames_grid.png` |

Sample frames were extracted from Safe, Caution, and High Risk classes. Some individual frames from different classes appeared visually similar. This supports the project direction of using temporal models instead of relying only on single-frame classification.

## Dataset Quality Assessment

The dataset quality assessment showed that the downloaded subset is usable for Milestone 2 preprocessing.

| Quality Check | Result |
|---|---|
| Missing videos | 0 |
| Videos failed to open | 0 |
| Class imbalance | Not observed in current subset |
| Mixed file formats | Present |
| Resolution variation | Present |
| FPS variation | Present |
| Visual similarity across individual frames | Present |

The main quality concern is that fatigue behavior may not always be visible in a single frame. Therefore, the model should learn from video sequences.

## Train / Validation / Test Split Strategy

The dataset was split at the subject level to prevent data leakage.

Random frame-level splitting was not used because frames from the same video are visually similar. If frames from the same video are placed in both training and testing sets, the model may learn video-specific patterns instead of general fatigue behavior.

| Split | Subject IDs | Number of Subjects | Number of Videos |
|---|---|---:|---:|
| Train | 01, 02, 03, 04, 05, 06, 07, 08 | 8 | 24 |
| Validation | 09, 10 | 2 | 6 |
| Test | 11, 12 | 2 | 6 |

## Class Balance After Split

| Split | Safe Videos | Caution Videos | High Risk Videos | Total Videos |
|---|---:|---:|---:|---:|
| Train | 8 | 8 | 8 | 24 |
| Validation | 2 | 2 | 2 | 6 |
| Test | 2 | 2 | 2 | 6 |

Since each subject contains one video per class, the split remains balanced.

## Leakage Prevention

The main leakage prevention rule is:

```text
No subject should appear in more than one split.
```

Frames and processed samples are generated only after the subject-level split is finalized. This ensures that frames or sequences from the same subject/video do not appear across train, validation, and test sets.

## Preprocessing Steps

The preprocessing pipeline includes the following steps:

1. Scan raw dataset folders.
2. Map file labels `0`, `5`, and `10` to Safe, Caution, and High Risk.
3. Support both `.mov` and `.mp4` video formats.
4. Extract video metadata using OpenCV.
5. Generate EDA charts.
6. Extract sample frames for visual inspection.
7. Create subject-level train/validation/test split.
8. Create processed dataset folder structure.
9. Extract controlled sample frames into class-wise folders.

## Processed Dataset Structure

The processed dataset is organized as follows:

```text
data/
  processed/
    fatigue_detection/
      train/
        Safe/
        Caution/
        High_Risk/
      val/
        Safe/
        Caution/
        High_Risk/
      test/
        Safe/
        Caution/
        High_Risk/
```

## Processed Sample Dataset

For Milestone 2, a small processed sample dataset was created by extracting five frames from each video at fixed positions:

```text
10%, 30%, 50%, 70%, 90%
```

This creates a lightweight processed dataset while proving that the pipeline can convert raw videos into model-ready folders.

| Split | Safe Frames | Caution Frames | High Risk Frames | Total Frames |
|---|---:|---:|---:|---:|
| Train | 40 | 40 | 40 | 120 |
| Validation | 10 | 10 | 10 | 30 |
| Test | 10 | 10 | 10 | 30 |
| **Total** | **60** | **60** | **60** | **180** |

## Dataset Adequacy

The selected UTA-RLDD subset is adequate for Milestone 2 because it supports:

- video-based fatigue detection
- three-class driver state classification
- subject-level splitting
- frame extraction
- metadata analysis
- temporal model preparation
- future sequence creation for CNN-LSTM, CNN-GRU, TCN, and lightweight 3D CNN

The dataset is also useful because the three classes naturally map to the project output levels: Safe, Caution, and High Risk.

## Limitations

The current work uses only `Fold1_part1` and `Fold1_part2` because the complete dataset is large. More folds can be added later using the same pipeline.

Other limitations include:

- Mixed video formats
- Multiple resolutions
- FPS variation
- Some visually similar frames across classes
- Large raw dataset size
- Need for temporal sequence modeling in M3

## Model Readiness for Milestone 3

The current processed dataset is ready for initial baseline experiments and visual inspection. For Milestone 3, the next step will be to create fixed-length frame sequences, such as 16-frame, 32-frame, or 64-frame clips.

These sequences can then be used with temporal models such as CNN-LSTM, CNN-GRU, TCN, or lightweight 3D CNN.
