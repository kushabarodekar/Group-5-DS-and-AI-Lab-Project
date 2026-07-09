from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SAMPLE_FRAMES_DIR = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/outputs/milestone-2/fatigue_sample_frames")
OUTPUT_DIR = Path("/Users/kushagra.barodekar/Learning/IITM/DS_And_AI_Lab/Milestone-2/outputs/milestone-2/fatigue_dataset_eda_charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "09_sample_frames_grid.png"

classes = [
    ("Safe", "Safe / Alert"),
    ("Caution", "Caution / Low Vigilance"),
    ("High_Risk", "High Risk / Drowsy")
]

images_per_class = 3
thumb_width = 320
thumb_height = 180
padding = 20
title_height = 50
label_height = 35

grid_width = images_per_class * thumb_width + (images_per_class + 1) * padding
grid_height = len(classes) * (thumb_height + label_height + padding) + title_height + padding

canvas = Image.new("RGB", (grid_width, grid_height), "white")
draw = ImageDraw.Draw(canvas)

try:
    title_font = ImageFont.truetype("Arial.ttf", 24)
    label_font = ImageFont.truetype("Arial.ttf", 18)
except:
    title_font = ImageFont.load_default()
    label_font = ImageFont.load_default()

title = "UTA-RLDD Sample Frames by Project Class"
draw.text((padding, 15), title, fill="black", font=title_font)

y = title_height + padding

for folder_name, label in classes:
    class_dir = SAMPLE_FRAMES_DIR / folder_name
    image_paths = sorted(class_dir.glob("*.jpg"))[:images_per_class]

    draw.text((padding, y), label, fill="black", font=label_font)
    image_y = y + label_height

    for idx, image_path in enumerate(image_paths):
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((thumb_width, thumb_height))

        thumb = Image.new("RGB", (thumb_width, thumb_height), "white")
        x_offset = (thumb_width - img.width) // 2
        y_offset = (thumb_height - img.height) // 2
        thumb.paste(img, (x_offset, y_offset))

        x = padding + idx * (thumb_width + padding)
        canvas.paste(thumb, (x, image_y))

    y += thumb_height + label_height + padding

canvas.save(OUTPUT_FILE)

print("Sample frame grid created.")
print(f"Saved at: {OUTPUT_FILE}")
