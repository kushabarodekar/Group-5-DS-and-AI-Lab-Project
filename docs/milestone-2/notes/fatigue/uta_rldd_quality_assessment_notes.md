# UTA-RLDD Dataset Quality Assessment Notes

## Dataset Reviewed

The current quality assessment was performed on the downloaded portions of the UTA Real-Life Drowsiness Dataset / UTA-RLDD.

Currently reviewed folders:

| Folder | Status |
|---|---|
| Fold1_part1 | Reviewed |
| Fold1_part2 | Reviewed |

## Raw Dataset Structure Quality

The dataset is organized subject-wise. Each subject folder contains three videos corresponding to different drowsiness levels.

| File Name | Dataset Class | Project Class |
|---|---|---|
| `0.mov` / `0.mp4` | Alert / Awake | Safe |
| `5.mov` / `5.mp4` | Low Vigilance | Caution |
| `10.mov` / `10.MOV` / `10.mp4` | Drowsy | High Risk |

This structure is suitable for creating a three-class video fatigue detection dataset.

## File Availability Check

All inspected subject folders contained the required three videos.

| Check | Result |
|---|---:|
| Subject folders inspected | 12 |
| Expected videos per subject | 3 |
| Total expected videos | 36 |
| Total videos found | 36 |
| Missing videos | 0 |

## Video Format Check

The dataset contains mixed video formats.

| Format | Observation |
|---|---|
| `.mov` / `.MOV` | Present in multiple subject folders |
| `.mp4` | Present in some subject folders |

Since both formats are valid video formats, preprocessing scripts must support both `.mov` and `.mp4`.

## Video Opening Check

All currently scanned videos were checked using OpenCV.

| Check | Result |
|---|---:|
| Total videos scanned | 36 |
| Videos opened successfully | 36 |
| Videos failed to open | 0 |

## Class Balance Check

For the currently downloaded folders, the dataset is balanced.

| Project Class | Number of Videos |
|---|---:|
| Safe | 12 |
| Caution | 12 |
| High Risk | 12 |

This is useful for model training because no class is currently underrepresented in the downloaded subset.

## Subject-Level Split Quality

The current split is subject-wise, which helps prevent leakage.

| Split | Subjects | Number of Subjects | Number of Videos |
|---|---|---:|---:|
| Train | 01, 02, 03, 04, 05, 06, 07, 08 | 8 | 24 |
| Validation | 09, 10 | 2 | 6 |
| Test | 11, 12 | 2 | 6 |

No subject appears in more than one split.

## Visual Quality Observation

Sample frames were extracted from all three project classes. Some isolated frames from Safe, Caution, and High Risk appear visually similar. This is expected because driver fatigue is not always visible in a single frame.

This observation supports the use of temporal deep learning models such as CNN-LSTM, CNN-GRU, TCN, or lightweight 3D CNN, because these models can learn changes in driver behavior across time.

## Possible Dataset Quality Issues

The following quality issues should still be considered during preprocessing:

- Some videos may contain visually similar frames across classes.
- Drowsiness cues may appear only during certain parts of the video.
- Some frames may not clearly show eyes or mouth.
- Some videos may have lighting variation.
- The dataset contains mixed file formats, so scripts must handle both `.mov` and `.mp4`.
- Since videos are long, frame extraction should be controlled to avoid unnecessary storage growth.

## Dataset Adequacy

The currently downloaded portion of UTA-RLDD is adequate for building the initial video-based fatigue detection pipeline. It provides subject-wise videos with three fatigue levels: Alert/Awake, Low Vigilance, and Drowsy.

The dataset is suitable for Milestone 2 because it allows:

- video metadata extraction
- class distribution analysis
- sample frame extraction
- frame sequence creation
- subject/video-wise train-validation-test splitting
- temporal model preparation for Milestone 3

## Notes for Milestone 2 Report

The visual similarity between individual sample frames should be mentioned as a reason for using video-sequence-based modeling instead of single-frame classification.
