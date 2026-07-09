from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

METADATA_CSV = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/outputs/milestone-2/fatigue_dataset_metadata/uta_rldd_video_metadata.csv")

metadata_df = pd.read_csv(METADATA_CSV)

output_dir = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/outputs/milestone-2/fatigue_dataset_eda_charts")
output_dir.mkdir(parents=True, exist_ok=True)

# Simple color palettes for better report visuals
BAR_COLORS = [
    "#4C78A8",  # blue
    "#F58518",  # orange
    "#54A24B",  # green
    "#E45756",  # red
    "#72B7B2",  # teal
    "#B279A2",  # purple
    "#FF9DA6",  # pink
    "#9D755D",  # brown
]

HIST_COLOR = "#4C78A8"


def save_bar_chart(series, title, xlabel, ylabel, filename):
    plt.figure(figsize=(9, 5.5))

    colors = BAR_COLORS[:len(series)]
    ax = series.plot(kind="bar", color=colors, edgecolor="black", linewidth=0.7)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, label_type="edge", fontsize=10, padding=3)

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_hist_chart(data, title, xlabel, ylabel, filename, bins=10):
    plt.figure(figsize=(9, 5.5))

    plt.hist(
        data.dropna(),
        bins=bins,
        color=HIST_COLOR,
        edgecolor="black",
        alpha=0.85
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


# 1. Class distribution
class_counts = metadata_df["project_class"].value_counts()
save_bar_chart(
    class_counts,
    "UTA-RLDD Class Distribution",
    "Project Class",
    "Number of Videos",
    "01_class_distribution.png"
)

# 2. Dataset class distribution
dataset_class_counts = metadata_df["dataset_class"].value_counts()
save_bar_chart(
    dataset_class_counts,
    "UTA-RLDD Dataset Class Distribution",
    "Dataset Class",
    "Number of Videos",
    "02_dataset_class_distribution.png"
)

# 3. File extension distribution
extension_counts = metadata_df["file_extension"].str.lower().value_counts()
save_bar_chart(
    extension_counts,
    "Video File Format Distribution",
    "File Extension",
    "Number of Videos",
    "03_file_format_distribution.png"
)

# 4. Resolution distribution
resolution_counts = metadata_df["resolution"].value_counts(dropna=False)
save_bar_chart(
    resolution_counts,
    "Video Resolution Distribution",
    "Resolution",
    "Number of Videos",
    "04_resolution_distribution.png"
)

# 5. FPS distribution
fps_counts = metadata_df["fps"].value_counts(dropna=False).sort_index()
save_bar_chart(
    fps_counts,
    "FPS Distribution",
    "FPS",
    "Number of Videos",
    "05_fps_distribution.png"
)

# 6. Duration distribution
save_hist_chart(
    metadata_df["duration_minutes"],
    "Video Duration Distribution",
    "Duration in Minutes",
    "Number of Videos",
    "06_duration_distribution.png",
    bins=10
)

# 7. Frame count distribution
save_hist_chart(
    metadata_df["frame_count"],
    "Frame Count Distribution",
    "Frame Count",
    "Number of Videos",
    "07_frame_count_distribution.png",
    bins=10
)

# 8. File size distribution
save_hist_chart(
    metadata_df["file_size_mb"],
    "File Size Distribution",
    "File Size in MB",
    "Number of Videos",
    "08_file_size_distribution.png",
    bins=10
)

print("EDA charts generated successfully.")
print(f"Charts saved in: {output_dir}")

print("\nGenerated files:")
for file in sorted(output_dir.glob("*.png")):
    print(file.name)