from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "dataset" / "UTA-RLDD"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "milestone-3"
MODEL_OUTPUT_ROOT = OUTPUT_ROOT / "model_outputs"
CANDIDATE_OUTPUT_ROOT = PROJECT_ROOT / "data" / "candidate_subset" / "fatigue_detection_sequences" / "seq_16"

CANDIDATE_VIDEO_SUBSET_CSV = PROJECT_ROOT / "scripts" / "uta_rldd_candidate_video_subset.csv"

SEQUENCE_LENGTH = 16
SEQUENCE_STRIDE = 8
TARGET_FPS = 5
FRAME_SIZE = (224, 224)

valid_video_extensions = {".mov", ".mp4"}

MODEL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CANDIDATE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

for split_name in ["train", "val", "test"]:
    for class_name in ["Safe", "Caution", "High_Risk"]:
        (CANDIDATE_OUTPUT_ROOT / split_name / class_name).mkdir(parents=True, exist_ok=True)


def class_to_folder_name(project_class: str) -> str:
    if project_class == "High Risk":
        return "High_Risk"
    return project_class


def extract_sampled_frames_from_video(video_path: Path, target_fps=5, frame_size=(224, 224)):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if not original_fps or original_fps <= 0:
        cap.release()
        return []

    frame_interval = max(int(round(original_fps / target_fps)), 1)

    sampled_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, frame_size)
            sampled_frames.append(frame)

        frame_idx += 1

    cap.release()
    return sampled_frames


def build_sequences_from_frames(frames, sequence_length=16, stride=8):
    sequences = []

    if len(frames) < sequence_length:
        return sequences

    for start_idx in range(0, len(frames) - sequence_length + 1, stride):
        seq = frames[start_idx:start_idx + sequence_length]
        seq = np.stack(seq, axis=0)
        sequences.append(seq)

    return sequences


def find_local_video_path(row):
    fold_name = str(row["fold"])
    subject_id = str(row["subject_id"]).zfill(2)
    label_code = str(row["label_code"])
    file_extension = str(row["file_extension"]).lower()
    video_file = str(row["video_file"])

    subject_dir = DATASET_ROOT / fold_name / subject_id

    # Try exact file name first
    exact_path = subject_dir / video_file
    if exact_path.exists():
        return exact_path

    # Fallback: stem match by label code
    if subject_dir.exists():
        for f in subject_dir.iterdir():
            if f.is_file() and f.suffix.lower() in valid_video_extensions and f.stem.lower() == label_code.lower():
                return f

    return None


def save_sequence(sequence, output_path: Path):
    np.save(output_path, sequence)


candidate_videos_df = pd.read_csv(CANDIDATE_VIDEO_SUBSET_CSV)

sequence_records = []

for _, row in candidate_videos_df.iterrows():
    subject_id = str(row["subject_id"]).zfill(2)
    split_name = row["split"]
    project_class = row["project_class"]
    folder_class = class_to_folder_name(project_class)
    label_code = str(row["label_code"])
    fold_name = row["fold"]

    video_path = find_local_video_path(row)

    if video_path is None or not video_path.exists():
        print(f"Skipping missing video for fold={fold_name}, subject={subject_id}, label={label_code}")
        continue

    frames = extract_sampled_frames_from_video(
        video_path,
        target_fps=TARGET_FPS,
        frame_size=FRAME_SIZE
    )

    sequences = build_sequences_from_frames(
        frames,
        sequence_length=SEQUENCE_LENGTH,
        stride=SEQUENCE_STRIDE
    )

    for seq_idx, seq in enumerate(sequences):
        seq_filename = f"{fold_name}_S{subject_id}_L{label_code}_SEQ{seq_idx:04d}.npy"
        seq_output_path = CANDIDATE_OUTPUT_ROOT / split_name / folder_class / seq_filename
        save_sequence(seq, seq_output_path)

        sequence_records.append({
            "fold": fold_name,
            "subject_id": subject_id,
            "split": split_name,
            "label_code": label_code,
            "project_class": project_class,
            "sequence_file": seq_filename,
            "sequence_path": str(seq_output_path.relative_to(PROJECT_ROOT)),
            "sequence_length": SEQUENCE_LENGTH,
            "frame_height": FRAME_SIZE[0],
            "frame_width": FRAME_SIZE[1],
            "target_fps": TARGET_FPS
        })

candidate_sequence_df = pd.DataFrame(sequence_records)

candidate_sequence_csv = MODEL_OUTPUT_ROOT / "uta_rldd_candidate_sequence_records.csv"
candidate_sequence_df.to_csv(candidate_sequence_csv, index=False)

print("Candidate sequence generation completed.")
print(f"Saved CSV: {candidate_sequence_csv}")
print(f"Total candidate sequences generated: {len(candidate_sequence_df)}")

if not candidate_sequence_df.empty:
    print("\nCandidate sequence distribution by split:")
    print(candidate_sequence_df["split"].value_counts())

    print("\nCandidate sequence distribution by split and class:")
    print(candidate_sequence_df.groupby(["split", "project_class"]).size())
