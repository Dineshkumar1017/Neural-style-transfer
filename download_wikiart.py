from datasets import load_dataset
from pathlib import Path
from PIL import Image
import io

out_dir = Path(r"C:\Users\DINESH KUMAR\Desktop\NST\wikiart_style")
out_dir.mkdir(parents=True, exist_ok=True)

print("Downloading WikiArt dataset from Hugging Face: voidik/wikiart")
ds = load_dataset("voidik/wikiart", split="train")
print(f"Rows found: {len(ds)}")

saved = 0
for i, row in enumerate(ds):
    image = row.get("image")
    try:
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, dict) and isinstance(image.get("bytes"), bytes):
            img = Image.open(io.BytesIO(image["bytes"]))
        elif isinstance(image, dict) and image.get("path"):
            img = Image.open(image["path"])
        else:
            continue

        img.convert("RGB").save(out_dir / f"wikiart_{i:05d}.jpg", quality=95)
        saved += 1
        if saved % 500 == 0:
            print(f"Saved {saved} images...")
    except Exception as exc:
        print(f"Skipped row {i}: {exc}")

print(f"Done. Saved {saved} images to {out_dir}")
