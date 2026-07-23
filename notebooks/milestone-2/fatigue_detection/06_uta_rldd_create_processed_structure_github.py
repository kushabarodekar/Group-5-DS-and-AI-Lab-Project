from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INVENTORY_CSV = PROJECT_ROOT / "outputs" / "milestone-3" / "fatigue_dataset_inventory" / "uta_rldd_video_inventory.csv"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "fatigue_detection"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "milestone-3" / "fatigue_dataset_splits"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

CLASSES = {
    "Safe": "0",
    "Caution": "5",
    "High_Risk": "10",
}


def main() -> None:
    if not INVENTORY_CSV.exists():
        raise FileNotFoundError(f"Inventory CSV not found: {INVENTORY_CSV}")

    inventory_df = pd.read_csv(INVENTORY_CSV)
    subject_ids = sorted(inventory_df["subject_id"].astype(str).str.zfill(2).unique())

    n_subjects = len(subject_ids)
    train_end = int(n_subjects * TRAIN_RATIO)
    val_end = int(n_subjects * (TRAIN_RATIO + VAL_RATIO))

    split_subjects = {
        "train": subject_ids[:train_end],
        "val": subject_ids[train_end:val_end],
        "test": subject_ids[val_end:],
    }

    records = []

    for split_name, subjects in split_subjects.items():
        for class_name in CLASSES:
            class_dir = PROCESSED_ROOT / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

        for subject_id in subjects:
            records.append({
                "split": split_name,
                "subject_id": subject_id,
            })

    split_df = pd.DataFrame(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_csv_path = OUTPUT_DIR / "uta_rldd_subject_split.csv"
    split_df.to_csv(split_csv_path, index=False)

    print("Processed dataset folder structure created.")
    print(f"Processed root: {PROCESSED_ROOT}")
    print(f"Subject split CSV: {split_csv_path}")

    print("\nTotal subjects:", n_subjects)
    print("Train subjects:", len(split_subjects["train"]))
    print("Val subjects:", len(split_subjects["val"]))
    print("Test subjects:", len(split_subjects["test"]))

    print("\nSubjects by split:")
    print(split_df.groupby("split")["subject_id"].apply(list))


if __name__ == "__main__":
    main()
