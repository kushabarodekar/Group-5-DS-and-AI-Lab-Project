from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FRAMES_DIR = PROJECT_ROOT / "outputs" / "milestone-3" / "fatigue_sample_frames"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "milestone-3" / "fatigue_dataset_eda_charts"
OUTPUT_FILE = OUTPUT_DIR / "10_sample_frames_grid.png"

CLASSES = [
    ("Safe", "Safe / Alert"),
    ("Caution", "Caution / Low Vigilance"),
    ("High_Risk", "High Risk / Drowsy"),
]

IMAGES_PER_CLASS = 4
THUMB_WIDTH = 320
THUMB_HEIGHT = 180
PADDING = 20
TITLE_HEIGHT = 50
LABEL_HEIGHT = 35


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid_width = IMAGES_PER_CLASS * THUMB_WIDTH + (IMAGES_PER_CLASS + 1) * PADDING
    grid_height = len(CLASSES) * (THUMB_HEIGHT + LABEL_HEIGHT + PADDING) + TITLE_HEIGHT + PADDING

    canvas = Image.new("RGB", (grid_width, grid_height), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("Arial.ttf", 24)
        label_font = ImageFont.truetype("Arial.ttf", 18)
    except Exception:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    title = "UTA-RLDD Full Dataset Sample Frames by Project Class"
    draw.text((PADDING, 15), title, fill="black", font=title_font)

    y = TITLE_HEIGHT + PADDING

    for folder_name, label in CLASSES:
        class_dir = SAMPLE_FRAMES_DIR / folder_name
        image_paths = sorted(class_dir.glob("*.jpg"))[:IMAGES_PER_CLASS]

        draw.text((PADDING, y), label, fill="black", font=label_font)
        image_y = y + LABEL_HEIGHT

        for idx, image_path in enumerate(image_paths):
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((THUMB_WIDTH, THUMB_HEIGHT))

            thumb = Image.new("RGB", (THUMB_WIDTH, THUMB_HEIGHT), "white")
            x_offset = (THUMB_WIDTH - img.width) // 2
            y_offset = (THUMB_HEIGHT - img.height) // 2
            thumb.paste(img, (x_offset, y_offset))

            x = PADDING + idx * (THUMB_WIDTH + PADDING)
            canvas.paste(thumb, (x, image_y))

        y += THUMB_HEIGHT + LABEL_HEIGHT + PADDING

    canvas.save(OUTPUT_FILE)

    print("Sample frame grid created.")
    print(f"Saved at: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
