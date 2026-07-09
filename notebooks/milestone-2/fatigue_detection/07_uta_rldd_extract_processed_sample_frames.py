from pathlib import Path
import shutil
import cv2
import pandas as pd

INVENTORY_CSV = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/outputs/milestone-2/fatigue_dataset_inventory/uta_rldd_video_inventory.csv")
SPLIT_CSV = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/outputs/milestone-2/fatigue_dataset_splits/uta_rldd_subject_split.csv")
PROCESSED_ROOT = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/data/processed/fatigue_detection")

inventory_df = pd.read_csv(INVENTORY_CSV)
split_df = pd.read_csv(SPLIT_CSV)

subject_to_split = dict(zip(split_df["subject_id"].astype(str).str.zfill(2), split_df["split"]))

class_folder_map = {
    "Safe": "Safe",
    "Caution": "Caution",
    "High Risk": "High_Risk"
}

# IMPORTANT:
# Clear existing processed sample frames before re-running.
# This prevents old frames from the previous split from remaining in the wrong folder.
if PROCESSED_ROOT.exists():
    for split_name in ["train", "val", "test"]:
        for class_folder in ["Safe", "Caution", "High_Risk"]:
            folder = PROCESSED_ROOT / split_name / class_folder
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)

# Controlled sample for M2.
# Extracting only a few frames keeps the processed dataset small.
FRAME_RATIOS = [0.10, 0.30, 0.50, 0.70, 0.90]

summary_records = []

for _, row in inventory_df.iterrows():
    subject_id = str(row["subject_id"]).zfill(2)
    split_name = subject_to_split.get(subject_id)

    if split_name is None:
        print(f"Skipping subject {subject_id}: no split assigned")
        continue

    project_class = row["project_class"]
    class_folder = class_folder_map[project_class]

    video_path = Path(row["video_path"])

    if not video_path.exists():
        print(f"Missing video file: {video_path}")
        continue

    output_dir = PROCESSED_ROOT / split_name / class_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        continue

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_count = 0

    for ratio in FRAME_RATIOS:
        target_frame = int(frame_count * ratio)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        success, frame = cap.read()

        if success:
            output_name = (
                f"{row['fold']}_subject_{subject_id}_"
                f"label_{row['label_code']}_frame_{target_frame}.jpg"
            )
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), frame)
            saved_count += 1
        else:
            print(f"Failed to read frame {target_frame} from {video_path}")

    cap.release()

    summary_records.append({
        "fold": row["fold"],
        "subject_id": subject_id,
        "split": split_name,
        "project_class": project_class,
        "video_file": row["video_file"],
        "frames_saved": saved_count
    })

summary_df = pd.DataFrame(summary_records)

output_summary_dir = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/outputs/milestone-2/fatigue_processed_sample")
output_summary_dir.mkdir(parents=True, exist_ok=True)

summary_path = output_summary_dir / "uta_rldd_processed_sample_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("Processed sample frame extraction completed.")
print(f"Summary CSV: {summary_path}")

print("\nFrames saved by split and class:")
print(summary_df.groupby(["split", "project_class"])["frames_saved"].sum())

print("\nTotal frames saved:", summary_df["frames_saved"].sum())
