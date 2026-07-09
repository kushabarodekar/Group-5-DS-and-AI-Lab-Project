from pathlib import Path
import pandas as pd

PROCESSED_ROOT = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/data/processed/fatigue_detection")

split_subjects = {
    "train": ["01", "02", "03", "04", "05", "06", "07", "08"],
    "val": ["09", "10"],
    "test": ["11", "12"]
}

classes = {
    "Safe": "0",
    "Caution": "5",
    "High_Risk": "10"
}

records = []

for split_name, subjects in split_subjects.items():
    for class_name, label_code in classes.items():
        class_dir = PROCESSED_ROOT / split_name / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

    for subject_id in subjects:
        records.append({
            "split": split_name,
            "subject_id": subject_id
        })

split_df = pd.DataFrame(records)

output_dir = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/outputs/milestone-2/fatigue_dataset_splits")
output_dir.mkdir(parents=True, exist_ok=True)

split_csv_path = output_dir / "uta_rldd_subject_split.csv"
split_df.to_csv(split_csv_path, index=False)

print("Processed dataset folder structure created.")
print(f"Processed root: {PROCESSED_ROOT}")
print(f"Subject split CSV: {split_csv_path}")

print("\nSplit summary:")
print(split_df["split"].value_counts())

print("\nSubjects by split:")
print(split_df.groupby("split")["subject_id"].apply(list))
