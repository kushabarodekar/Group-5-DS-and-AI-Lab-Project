from pathlib import Path
import cv2
import pandas as pd

INVENTORY_CSV = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/outputs/milestone-2/fatigue_dataset_inventory/uta_rldd_video_inventory.csv")

inventory_df = pd.read_csv(INVENTORY_CSV)

metadata_records = []

for _, row in inventory_df.iterrows():
    video_path = Path(row["video_path"])

    record = {
        "fold": row["fold"],
        "subject_id": row["subject_id"],
        "video_file": row["video_file"],
        "label_code": row["label_code"],
        "dataset_class": row["dataset_class"],
        "project_class": row["project_class"],
        "file_extension": row["file_extension"],
        "file_size_mb": row["file_size_mb"],
        "video_path": str(video_path),
        "can_open": False,
        "fps": None,
        "frame_count": None,
        "duration_seconds": None,
        "duration_minutes": None,
        "width": None,
        "height": None,
        "resolution": None
    }

    cap = cv2.VideoCapture(str(video_path))

    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration_seconds = None
        duration_minutes = None

        if fps and fps > 0:
            duration_seconds = frame_count / fps
            duration_minutes = duration_seconds / 60

        record.update({
            "can_open": True,
            "fps": round(fps, 2) if fps else None,
            "frame_count": int(frame_count) if frame_count else None,
            "duration_seconds": round(duration_seconds, 2) if duration_seconds else None,
            "duration_minutes": round(duration_minutes, 2) if duration_minutes else None,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}"
        })

    cap.release()
    metadata_records.append(record)

metadata_df = pd.DataFrame(metadata_records)

output_dir = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/outputs/milestone-2/fatigue_dataset_metadata")
output_dir.mkdir(parents=True, exist_ok=True)

metadata_path = output_dir / "uta_rldd_video_metadata.csv"
metadata_df.to_csv(metadata_path, index=False)

print("Video metadata CSV created.")
print(f"Metadata CSV: {metadata_path}")

print("\nTotal videos scanned:", len(metadata_df))
print("Videos opened successfully:", metadata_df["can_open"].sum())
print("Videos failed to open:", len(metadata_df) - metadata_df["can_open"].sum())

print("\nClass distribution:")
print(metadata_df["project_class"].value_counts())

print("\nResolution distribution:")
print(metadata_df["resolution"].value_counts(dropna=False))

print("\nFPS distribution:")
print(metadata_df["fps"].value_counts(dropna=False).sort_index())

print("\nDuration summary in minutes:")
print(metadata_df["duration_minutes"].describe())
