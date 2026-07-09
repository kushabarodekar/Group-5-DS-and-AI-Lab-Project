# UTA-RLDD Processed Dataset Structure Notes

## Purpose

This file documents the processed dataset structure created for the video-based fatigue detection module.

The goal of this processed structure is to make the UTA-RLDD dataset ready for Milestone 3 model experimentation. The processed dataset is organized by train, validation, and test splits, and each split contains class-wise folders.

## Raw Dataset Used

The processed sample dataset was created from the currently downloaded UTA-RLDD subset:

| Fold | Subjects Included | Number of Subjects |
|---|---|---:|
| Fold1_part1 | 01, 02, 03, 04, 05, 06 | 6 |
| Fold1_part2 | 07, 08, 09, 10, 11, 12 | 6 |

Total subjects used: **12**

Each subject contains three videos:

| File Name | Dataset Class | Project Class |
|---|---|---|
| `0.mov` / `0.mp4` | Alert / Awake | Safe |
| `5.mov` / `5.mp4` | Low Vigilance | Caution |
| `10.mov` / `10.mp4` | Drowsy | High Risk |

Total videos used: **36**

## Subject-Level Split

The dataset was split at the subject level to avoid data leakage.

| Split | Subject IDs | Number of Subjects | Number of Videos |
|---|---|---:|---:|
| Train | 01, 02, 03, 04, 05, 06, 07, 08 | 8 | 24 |
| Validation | 09, 10 | 2 | 6 |
| Test | 11, 12 | 2 | 6 |

## Processed Folder Structure

The processed dataset is stored in the following structure:

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

## Processed Sample Frame Extraction

For Milestone 2, a controlled processed sample dataset was created by extracting five frames from each video.

Frames were extracted at the following video positions:

```text
10%, 30%, 50%, 70%, 90%
```

This keeps the processed dataset small while still proving that the pipeline can convert raw videos into a model-ready folder structure.

## Processed Frame Count

| Split | Safe Frames | Caution Frames | High Risk Frames | Total Frames |
|---|---:|---:|---:|---:|
| Train | 40 | 40 | 40 | 120 |
| Validation | 10 | 10 | 10 | 30 |
| Test | 10 | 10 | 10 | 30 |
| **Total** | **60** | **60** | **60** | **180** |

## Output Files Generated

The preprocessing pipeline generated the following output files:

| Output File | Purpose |
|---|---|
| `uta_rldd_video_inventory.csv` | Stores fold, subject, video path, file type, label code, dataset class, and project class. |
| `uta_rldd_missing_files_report.csv` | Stores missing expected video files, if any. |
| `uta_rldd_video_metadata.csv` | Stores FPS, duration, resolution, frame count, file size, and video open status. |
| `uta_rldd_subject_split.csv` | Stores subject-level train/validation/test assignment. |
| `uta_rldd_processed_sample_summary.csv` | Stores number of processed frames extracted per video. |

## Model Readiness

The processed dataset is ready for initial Milestone 3 experiments using image-based baseline models or frame-level visual inspection.

For temporal models such as CNN-LSTM, CNN-GRU, TCN, or lightweight 3D CNN, the next preprocessing step will be to group extracted frames into fixed-length sequences.

The current processed structure already supports this because the data is organized by split and class, and the subject-level split prevents data leakage.

## Important Notes

- The raw dataset is large, so only the processed sample output will be hosted for Milestone 2.
- Full raw videos will remain stored locally due to size limitations.
- The same preprocessing scripts can be applied to additional UTA-RLDD folds as they are downloaded.
- The dataset contains both `.mov` and `.mp4` files, so all scripts were updated to support both formats.
- Individual sample frames from different classes may look visually similar. This supports the decision to use temporal models instead of relying only on single-frame classification.
