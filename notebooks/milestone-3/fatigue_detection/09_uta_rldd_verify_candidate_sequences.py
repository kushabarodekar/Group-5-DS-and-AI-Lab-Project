from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_SEQUENCE_CSV = PROJECT_ROOT / "outputs" / "milestone-3" / "model_outputs" / "uta_rldd_candidate_sequence_records.csv"

candidate_sequence_df = pd.read_csv(CANDIDATE_SEQUENCE_CSV)

print("Loaded candidate sequence records:", len(candidate_sequence_df))

print("\nSequence distribution by split:")
print(candidate_sequence_df["split"].value_counts())

print("\nSequence distribution by split and class:")
print(candidate_sequence_df.groupby(["split", "project_class"]).size())

# Check whether files exist
candidate_sequence_df["file_exists"] = candidate_sequence_df["sequence_path"].apply(lambda x: (PROJECT_ROOT / Path(x)).exists())

print("\nExisting sequence files:", candidate_sequence_df["file_exists"].sum())
print("Missing sequence files:", len(candidate_sequence_df) - candidate_sequence_df["file_exists"].sum())

# Load one sample file from each split/class where possible
print("\nSample shape checks:")
for split_name in ["train", "val", "test"]:
    for project_class in ["Safe", "Caution", "High Risk"]:
        subset = candidate_sequence_df[
            (candidate_sequence_df["split"] == split_name) &
            (candidate_sequence_df["project_class"] == project_class)
        ]

        if subset.empty:
            print(f"{split_name} / {project_class}: no records")
            continue

        sample_path = PROJECT_ROOT / Path(subset.iloc[0]["sequence_path"])
        if not sample_path.exists():
            print(f"{split_name} / {project_class}: sample file missing -> {sample_path}")
            continue

        seq = np.load(sample_path)
        print(f"{split_name} / {project_class}: {seq.shape}")

# Save a small verification report
verification_output = PROJECT_ROOT / "outputs" / "milestone-3" / "model_outputs" / "uta_rldd_candidate_sequence_verification.csv"
candidate_sequence_df[["fold", "subject_id", "split", "project_class", "sequence_file", "file_exists"]].to_csv(
    verification_output,
    index=False
)

print(f"\nSaved verification report: {verification_output}")
