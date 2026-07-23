from pathlib import Path
import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_CSV = PROJECT_ROOT / "outputs" / "milestone-3" / "fatigue_dataset_metadata" / "uta_rldd_video_metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "milestone-3" / "fatigue_sample_frames"

VIDEOS_PER_CLASS = 6
FRAME_RATIOS = [0.20, 0.50, 0.80]


def main() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    metadata_df = pd.read_csv(METADATA_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for project_class, class_df in metadata_df.groupby("project_class"):
        class_folder_name = project_class.replace(" ", "_").replace("/", "_")
        class_output_dir = OUTPUT_DIR / class_folder_name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        # Spread selection across folds/subjects for better full-dataset representation.
        sampled_videos = (
            class_df.sort_values(["fold", "subject_id", "label_code"])
            .drop_duplicates(subset=["subject_id"], keep="first")
            .head(VIDEOS_PER_CLASS)
        )

        for _, row in sampled_videos.iterrows():
            video_path = PROJECT_ROOT / Path(row["video_path"])

            if not video_path.exists():
                print(f"Missing video file: {video_path}")
                continue

            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for ratio in FRAME_RATIOS:
                target_frame = int(frame_count * ratio)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

                success, frame = cap.read()

                if success:
                    output_name = (
                        f"{row['fold']}_subject_{str(row['subject_id']).zfill(2)}_"
                        f"label_{row['label_code']}_frame_{target_frame}.jpg"
                    )
                    output_path = class_output_dir / output_name
                    cv2.imwrite(str(output_path), frame)
                else:
                    print(f"Failed to read frame {target_frame} from {video_path}")

            cap.release()

    print("Sample frame extraction completed.")
    print(f"Sample frames saved in: {OUTPUT_DIR}")

    for class_dir in sorted(OUTPUT_DIR.iterdir()):
        if class_dir.is_dir():
            print(f"{class_dir.name}: {len(list(class_dir.glob('*.jpg')))} frames")


if __name__ == "__main__":
    main()
