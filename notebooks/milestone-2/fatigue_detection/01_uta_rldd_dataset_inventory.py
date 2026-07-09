from pathlib import Path
import pandas as pd

# Change this path based on where your UTA-RLDD folder is stored
DATASET_ROOT = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/data/raw/fatigue_detection/UTA-RLDD")

label_map = {
    "0": {
        "dataset_class": "Alert / Awake",
        "project_class": "Safe"
    },
    "5": {
        "dataset_class": "Low Vigilance",
        "project_class": "Caution"
    },
    "10": {
        "dataset_class": "Drowsy",
        "project_class": "High Risk"
    }
}

valid_video_extensions = {".mov", ".mp4"}

records = []
missing_records = []

for fold_dir in sorted(DATASET_ROOT.iterdir()):
    if not fold_dir.is_dir():
        continue

    for subject_dir in sorted(fold_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name

        existing_files = {
            file.stem.lower(): file
            for file in subject_dir.iterdir()
            if file.is_file() and file.suffix.lower() in valid_video_extensions
        }

        for expected_label in ["0", "5", "10"]:
            if expected_label in existing_files:
                video_path = existing_files[expected_label]
                file_size_mb = video_path.stat().st_size / (1024 * 1024)

                records.append({
                    "fold": fold_dir.name,
                    "subject_id": subject_id,
                    "video_file": video_path.name,
                    "label_code": expected_label,
                    "dataset_class": label_map[expected_label]["dataset_class"],
                    "project_class": label_map[expected_label]["project_class"],
                    "file_extension": video_path.suffix,
                    "file_size_mb": round(file_size_mb, 2),
                    "video_path": str(video_path)
                })
            else:
                missing_records.append({
                    "fold": fold_dir.name,
                    "subject_id": subject_id,
                    "missing_label_code": expected_label,
                    "expected_dataset_class": label_map[expected_label]["dataset_class"],
                    "expected_project_class": label_map[expected_label]["project_class"]
                })

inventory_df = pd.DataFrame(records)
missing_df = pd.DataFrame(missing_records)

output_dir = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/outputs/milestone-2/fatigue_dataset_inventory")
output_dir.mkdir(parents=True, exist_ok=True)

inventory_path = output_dir / "uta_rldd_video_inventory.csv"
missing_path = output_dir / "uta_rldd_missing_files_report.csv"

inventory_df.to_csv(inventory_path, index=False)
missing_df.to_csv(missing_path, index=False)

print("Dataset inventory created.")
print(f"Inventory CSV: {inventory_path}")
print(f"Missing files CSV: {missing_path}")

print("\nTotal videos found:", len(inventory_df))
print("Total missing expected files:", len(missing_df))

if not inventory_df.empty:
    print("\nVideos per class:")
    print(inventory_df["project_class"].value_counts())

    print("\nSubjects per fold:")
    print(inventory_df.groupby("fold")["subject_id"].nunique())

    print("\nFile extensions:")
    print(inventory_df["file_extension"].str.lower().value_counts())
