from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_CSV = PROJECT_ROOT / "outputs" / "milestone-3" / "fatigue_dataset_metadata" / "uta_rldd_video_metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "milestone-3" / "fatigue_dataset_eda_charts"

BAR_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#F2CF5B",
]

HIST_COLOR = "#4C78A8"


def save_bar_chart(series, title, xlabel, ylabel, filename, rotation=30):
    plt.figure(figsize=(10, 5.5))
    colors = BAR_COLORS[:len(series)]
    ax = series.plot(kind="bar", color=colors, edgecolor="black", linewidth=0.7)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.xticks(rotation=rotation, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for container in ax.containers:
        ax.bar_label(container, label_type="edge", fontsize=9, padding=3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_hist_chart(data, title, xlabel, ylabel, filename, bins=12):
    plt.figure(figsize=(10, 5.5))
    plt.hist(data.dropna(), bins=bins, color=HIST_COLOR, edgecolor="black", alpha=0.85)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    metadata_df = pd.read_csv(METADATA_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_bar_chart(
        metadata_df["project_class"].value_counts().sort_index(),
        "UTA-RLDD Full Dataset Class Distribution",
        "Project Class",
        "Number of Videos",
        "01_class_distribution.png",
        rotation=0,
    )

    save_bar_chart(
        metadata_df["dataset_class"].value_counts().sort_index(),
        "UTA-RLDD Dataset Class Distribution",
        "Dataset Class",
        "Number of Videos",
        "02_dataset_class_distribution.png",
        rotation=0,
    )

    save_bar_chart(
        metadata_df["fold"].value_counts().sort_index(),
        "UTA-RLDD Fold Distribution",
        "Fold",
        "Number of Videos",
        "03_fold_distribution.png",
        rotation=45,
    )

    save_bar_chart(
        metadata_df["file_extension"].str.lower().value_counts().sort_index(),
        "Video File Format Distribution",
        "File Extension",
        "Number of Videos",
        "04_file_format_distribution.png",
        rotation=0,
    )

    save_bar_chart(
        metadata_df["resolution"].value_counts(dropna=False).head(15),
        "Top Video Resolution Distribution",
        "Resolution",
        "Number of Videos",
        "05_resolution_distribution.png",
        rotation=45,
    )

    save_bar_chart(
        metadata_df["fps"].value_counts(dropna=False).sort_index(),
        "FPS Distribution",
        "FPS",
        "Number of Videos",
        "06_fps_distribution.png",
        rotation=45,
    )

    save_hist_chart(
        metadata_df["duration_minutes"],
        "Video Duration Distribution",
        "Duration (minutes)",
        "Number of Videos",
        "07_duration_distribution.png",
        bins=12,
    )

    save_hist_chart(
        metadata_df["frame_count"],
        "Frame Count Distribution",
        "Frame Count",
        "Number of Videos",
        "08_frame_count_distribution.png",
        bins=12,
    )

    save_hist_chart(
        metadata_df["file_size_mb"],
        "File Size Distribution",
        "File Size (MB)",
        "Number of Videos",
        "09_file_size_distribution.png",
        bins=12,
    )

    print("EDA charts generated successfully.")
    print(f"Charts saved in: {OUTPUT_DIR}")

    print("\nGenerated files:")
    for file_path in sorted(OUTPUT_DIR.glob("*.png")):
        print(file_path.name)


if __name__ == "__main__":
    main()
