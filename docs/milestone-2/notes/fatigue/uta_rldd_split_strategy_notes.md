# UTA-RLDD Train / Validation / Test Split Strategy

## Dataset Split Requirement

For Milestone 2, the UTA-RLDD dataset must be prepared in a structure that can be directly used for model experiments in Milestone 3.

Since this is a video-based fatigue detection dataset, the split must be done carefully to avoid data leakage.

## Why Frame-Level Random Split Is Not Used

Random frame-level splitting is not suitable for this dataset because frames from the same video are highly similar. If frames from the same video are placed in both training and testing sets, the model may learn video-specific visual patterns instead of general fatigue behavior.

This can produce misleadingly high test accuracy.

To avoid this issue, the dataset will be split at the **subject level**.

## Subject-Level Split Strategy

All videos from the same subject will stay in only one split.

This means that if subject `01` is assigned to the training set, then all three videos from subject `01` will remain in the training set:

- `0.mov` or `0.mp4`
- `5.mov` or `5.mp4`
- `10.mov`, `10.MOV`, or `10.mp4`

The same rule will be applied to validation and test sets.

## Current Downloaded Subjects

The currently downloaded and inspected subset contains the following subject folders:

```text
01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12
```

Total subjects currently available: **12**

Each subject has three videos:

- Alert / Awake
- Low Vigilance
- Drowsy

Total videos currently available: **36**

## Proposed Split for Current Downloaded Subset

| Split | Subject IDs | Number of Subjects | Number of Videos |
|---|---|---:|---:|
| Train | 01, 02, 03, 04, 05, 06, 07, 08 | 8 | 24 |
| Validation | 09, 10 | 2 | 6 |
| Test | 11, 12 | 2 | 6 |

## Class Balance After Split

Since each subject contains one video for each class, class balance is naturally maintained within each split.

| Split | Safe Videos | Caution Videos | High Risk Videos | Total Videos |
|---|---:|---:|---:|---:|
| Train | 8 | 8 | 8 | 24 |
| Validation | 2 | 2 | 2 | 6 |
| Test | 2 | 2 | 2 | 6 |

## Leakage Prevention

The main leakage prevention rule is:

```text
No subject should appear in more than one split.
```

Also:

```text
Frames or sequences should be generated only after the subject-level split is finalized.
```

This ensures that extracted frames or generated sequences from the same video do not appear across multiple splits.

## Processed Sample Dataset Plan

For Milestone 2, a small processed sample dataset will be created by extracting five frames from each video at fixed positions across the video timeline.

For the current subset:

```text
36 videos x 5 frames per video = 180 processed sample frames
```

The expected processed sample frame count is:

| Split | Safe Frames | Caution Frames | High Risk Frames | Total Frames |
|---|---:|---:|---:|---:|
| Train | 40 | 40 | 40 | 120 |
| Validation | 10 | 10 | 10 | 30 |
| Test | 10 | 10 | 10 | 30 |
| **Total** | **60** | **60** | **60** | **180** |

## Future Extension

As more UTA-RLDD fold parts are downloaded, the same subject-level split strategy will be applied. The split ratio can be adjusted closer to 70/15/15 when more subjects are available.

For the current Milestone 2 subset, the selected split is suitable for preparing a model-ready fatigue detection pipeline.
