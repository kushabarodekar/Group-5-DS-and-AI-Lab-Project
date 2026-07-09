# UTA-RLDD Raw Dataset Inventory Notes

## Download Status

The full UTA Real-Life Drowsiness Dataset / UTA-RLDD is large, approximately 96 GB, so the dataset is being downloaded and processed fold-wise.

Currently downloaded and extracted folders:

| Folder | Status |
|---|---|
| Fold1_part1 | Downloaded and extracted |
| Fold1_part2 | Downloaded and extracted |

## Observed Folder Structure

The dataset is organized by subject folders. Each subject folder contains three video files representing different drowsiness levels.

Observed structure:

```text
UTA-RLDD/
  Fold1_part1/
    01/
      0.mov
      5.mov
      10.MOV
    02/
      0.mov
      5.mov
      10.MOV
    03/
      0.mov
      5.mov
      10.MOV
    04/
      0.mp4
      5.mp4
      10.mp4
    05/
      0.mov
      5.mov
      10.MOV
    06/
      0.mov or 0.mp4
      5.mov or 5.mp4
      10.MOV or 10.mp4

  Fold1_part2/
    07/
      0.mp4
      5.mp4
      10.mp4
    08/
      0.mp4
      5.mp4
      10.mp4
    09/
      0.mp4
      5.mp4
      10.mp4
    10/
      0.mov
      5.mov
      10.MOV
    11/
      0.mp4
      5.mp4
      10.mp4
    12/
      0.mov or 0.mp4
      5.mov or 5.mp4
      10.MOV or 10.mp4
```

## Label Mapping

Based on the dataset file naming convention, each subject folder contains three videos:

| Video File Name | Dataset Class | Project Class |
|---|---|---|
| `0.mov` / `0.mp4` | Alert / Awake | Safe |
| `5.mov` / `5.mp4` | Low Vigilance | Caution |
| `10.mov` / `10.MOV` / `10.mp4` | Drowsy | High Risk |

This label mapping supports the project’s three-level driver wellness classification:

- **Safe**
- **Caution**
- **High Risk**

## Manual and Script-Based Check for Fold1_part1

The `Fold1_part1` folder was inspected manually and then verified through the dataset inventory script. It contains subject folders from `01` to `06`.

Each subject folder contains all three required video files.

| Subject Folder | `0` Video Present | `5` Video Present | `10` Video Present | File Format Observed | Notes |
|---|---|---|---|---|---|
| 01 | Yes | Yes | Yes | `.mov` / `.MOV` | All videos present |
| 02 | Yes | Yes | Yes | `.mov` / `.MOV` | All videos present |
| 03 | Yes | Yes | Yes | `.mov` / `.MOV` | All videos present |
| 04 | Yes | Yes | Yes | `.mp4` | All videos present |
| 05 | Yes | Yes | Yes | `.mov` / `.MOV` | All videos present |
| 06 | Yes | Yes | Yes | `.mov` / `.mp4` | All videos present |

## Manual and Script-Based Check for Fold1_part2

The `Fold1_part2` folder was inspected manually and then verified through the dataset inventory script. It contains subject folders from `07` to `12`.

Each subject folder contains all three required video files.

| Subject Folder | `0` Video Present | `5` Video Present | `10` Video Present | File Format Observed | Notes |
|---|---|---|---|---|---|
| 07 | Yes | Yes | Yes | `.mp4` | All videos present |
| 08 | Yes | Yes | Yes | `.mp4` | All videos present |
| 09 | Yes | Yes | Yes | `.mp4` | All videos present |
| 10 | Yes | Yes | Yes | `.mov` / `.MOV` | All videos present |
| 11 | Yes | Yes | Yes | `.mp4` | All videos present |
| 12 | Yes | Yes | Yes | `.mov` / `.mp4` | All videos present |

## Current Dataset Count

Based on the current inspection of `Fold1_part1` and `Fold1_part2`:

| Item | Count |
|---|---:|
| Downloaded fold parts inspected | 2 |
| Subject folders inspected | 12 |
| Videos per subject | 3 |
| Total videos inspected | 36 |
| Missing videos found | 0 |
| Video formats observed | `.mov`, `.MOV`, `.mp4` |
| Approximate duration per video | Around 10 minutes |

## Important Observation

The dataset contains mixed video formats. Some subject folders use `.mov` / `.MOV`, while others use `.mp4`.

Therefore, the dataset scanning and preprocessing scripts should support both video extensions:

```text
.mov
.MOV
.mp4
.MP4
```

The label should be identified using the file stem/name, such as `0`, `5`, or `10`, not only by file extension.

## Notes for Milestone 2

Since the full dataset is very large, the dataset will be processed in parts. The current EDA and preprocessing pipeline will be developed using `Fold1_part1` and `Fold1_part2` first. The same pipeline can then be applied to the remaining downloaded fold parts.

For Milestone 2, this approach helps verify the dataset structure, label mapping, file format variation, and preprocessing requirements before processing the complete dataset.

The current dataset subset is suitable for video-based fatigue detection because each subject has videos for all three fatigue states: alert/awake, low vigilance, and drowsy.
