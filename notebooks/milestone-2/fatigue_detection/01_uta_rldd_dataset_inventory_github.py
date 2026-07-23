from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "UTA-RLDD"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "milestone-3" / "fatigue_dataset_inventory"

LABEL_MAP = {
    "0": {"dataset_class": "Alert / Awake", "project_class": "Safe"},
    "5": {"dataset_class": "Low Vigilance", "project_class": "Caution"},
    "10": {"dataset_class": "Drowsy", "project_class": "High Risk"},
}

VALID_VIDEO_EXTENSIONS = {".mov", ".mp4"}


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    records = []
    missing_records = []

    fold_dirs = sorted([path for path in DATASET_ROOT.iterdir() if path.is_dir()])

    for fold_dir in fold_dirs:
        for subject_dir in sorted([path for path in fold_dir.iterdir() if path.is_dir()]):
            subject_id = subject_dir.name

            existing_files = {
                file_path.stem.lower(): file_path
                for file_path in subject_dir.iterdir()
                if file_path.is_file() and file_path.suffix.lower() in VALID_VIDEO_EXTENSIONS
            }

            for expected_label in ["0", "5", "10"]:
                if expected_label in existing_files:
                    video_path = existing_files[expected_label]
                    file_size_mb = video_path.stat().st_size / (1024 * 1024)

                    records.append({
                        "fold": fold_dir.name,
                        "subject_id": str(subject_id).zfill(2),
                        "video_file": video_path.name,
                        "label_code": expected_label,
                        "dataset_class": LABEL_MAP[expected_label]["dataset_class"],
                        "project_class": LABEL_MAP[expected_label]["project_class"],
                        "file_extension": video_path.suffix.lower(),
                        "file_size_mb": round(file_size_mb, 2),
                        "video_path": str(video_path.relative_to(PROJECT_ROOT)),
                    })
                else:
                    missing_records.append({
                        "fold": fold_dir.name,
                        "subject_id": str(subject_id).zfill(2),
                        "missing_label_code": expected_label,
                        "expected_dataset_class": LABEL_MAP[expected_label]["dataset_class"],
                        "expected_project_class": LABEL_MAP[expected_label]["project_class"],
                    })

    inventory_df = pd.DataFrame(records).sort_values(["fold", "subject_id", "label_code"]).reset_index(drop=True)
    missing_df = pd.DataFrame(missing_records).sort_values(["fold", "subject_id", "missing_label_code"]).reset_index(drop=True)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    inventory_path = OUTPUT_ROOT / "uta_rldd_video_inventory.csv"
    missing_path = OUTPUT_ROOT / "uta_rldd_missing_files_report.csv"

    inventory_df.to_csv(inventory_path, index=False)
    missing_df.to_csv(missing_path, index=False)

    print("Dataset inventory created.")
    print(f"Inventory CSV: {inventory_path}")
    print(f"Missing files CSV: {missing_path}")

    print("\nTotal folds found:", inventory_df["fold"].nunique() if not inventory_df.empty else 0)
    print("Total subjects found:", inventory_df["subject_id"].nunique() if not inventory_df.empty else 0)
    print("Total videos found:", len(inventory_df))
    print("Total missing expected files:", len(missing_df))

    if not inventory_df.empty:
        print("\nVideos per class:")
        print(inventory_df["project_class"].value_counts().sort_index())

        print("\nVideos per fold:")
        print(inventory_df.groupby("fold")["video_file"].count())

        print("\nSubjects per fold:")
        print(inventory_df.groupby("fold")["subject_id"].nunique())

        print("\nFile extensions:")
        print(inventory_df["file_extension"].value_counts())


if __name__ == "__main__":
    main()
