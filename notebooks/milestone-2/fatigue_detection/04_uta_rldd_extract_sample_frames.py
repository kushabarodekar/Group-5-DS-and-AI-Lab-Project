from pathlib import Path
import cv2
import pandas as pd

METADATA_CSV = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/outputs/milestone-2/fatigue_dataset_metadata/uta_rldd_video_metadata.csv")

metadata_df = pd.read_csv(METADATA_CSV)

output_dir = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/outputs/milestone-2/fatigue_sample_frames")
output_dir.mkdir(parents=True, exist_ok=True)

# Number of videos to sample per class
VIDEOS_PER_CLASS = 3

# Frame positions to extract from each selected video
# These are ratios of video duration: 25%, 50%, and 75%
FRAME_RATIOS = [0.25, 0.50, 0.75]

for project_class, class_df in metadata_df.groupby("project_class"):
    class_folder_name = project_class.replace(" ", "_").replace("/", "_")
    class_output_dir = output_dir / class_folder_name
    class_output_dir.mkdir(parents=True, exist_ok=True)

    # Select first few videos from each class
    sampled_videos = class_df.head(VIDEOS_PER_CLASS)

    for _, row in sampled_videos.iterrows():
        video_path = Path(row["video_path"])

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
                    f"{row['fold']}_subject_{row['subject_id']}_"
                    f"label_{row['label_code']}_frame_{target_frame}.jpg"
                )
                output_path = class_output_dir / output_name
                cv2.imwrite(str(output_path), frame)
            else:
                print(f"Failed to read frame {target_frame} from {video_path}")

        cap.release()

print("Sample frame extraction completed.")
print(f"Sample frames saved in: {output_dir}")

for class_dir in sorted(output_dir.iterdir()):
    if class_dir.is_dir():
        print(f"{class_dir.name}: {len(list(class_dir.glob('*.jpg')))} frames")
