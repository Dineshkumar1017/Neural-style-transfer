from datasets import load_dataset
from pathlib import Path
from PIL import Image
import io

out_dir = Path(r"C:\Users\DINESH KUMAR\Desktop\NST\wikiart_style")
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading WikiArt dataset from Hugging Face: voidik/wikiart")
ds = load_dataset("voidik/wikiart", split="train")
print(f"Rows found: {len(ds)}")

saved = 0
skipped_existing = 0
skipped_error = 0

for i, row in enumerate(ds):
    output_path = out_dir / f"wikiart_{i:05d}.jpg"
    if output_path.exists() and output_path.stat().st_size > 0:
        skipped_existing += 1
        continue

    image = row.get("image")
    try:
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, dict) and isinstance(image.get("bytes"), bytes):
            img = Image.open(io.BytesIO(image["bytes"]))
        elif isinstance(image, dict) and image.get("path"):
            img = Image.open(image["path"])
        else:
            skipped_error += 1
            continue

        img.convert("RGB").save(output_path, quality=95)
        saved += 1
        if saved % 500 == 0:
            print(f"Saved {saved} new images... existing skipped: {skipped_existing}")
    except Exception as exc:
        skipped_error += 1
        print(f"Skipped row {i}: {exc}")

print(f"Done. Saved {saved} new images. Existing skipped: {skipped_existing}. Errors: {skipped_error}.")
print(f"Output folder: {out_dir}")
