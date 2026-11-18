"""
Download 30 real shirt product images from a CC0 clothing-image dataset.

Source repository:
https://github.com/lightly-ai/dataset_clothing_images
The repository identifies its license as CC0-1.0.

Run:
    python download_30_shirts.py

Output:
    shirt_images_30/
        shirt_001.jpg ...
        source_manifest.txt
    shirt_images_30.zip

The script samples from the repository's shirt, t-shirt and longsleeve
categories, aiming for a varied mix. It does not scrape retail product pages.
"""

from pathlib import Path
from urllib.request import Request, urlopen
import json
import random
import time
import zipfile

API_URL = (
    "https://api.github.com/repos/lightly-ai/"
    "dataset_clothing_images/git/trees/main?recursive=1"
)
RAW_BASE = (
    "https://raw.githubusercontent.com/lightly-ai/"
    "dataset_clothing_images/main/"
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "shirt_images_30"
ZIP_PATH = ROOT / "shirt_images_30.zip"
TARGET = 30
RANDOM_SEED = 42

CATEGORY_FOLDERS = ("shirt", "t-shirt", "longsleeve")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def fetch_bytes(url):
    request = Request(
        url,
        headers={"User-Agent": "EyeTheia-shirt-image-downloader/1.0"}
    )
    with urlopen(request, timeout=45) as response:
        return response.read()


def main():
    print("Fetching the public clothing dataset file list...")
    tree = json.loads(fetch_bytes(API_URL).decode("utf-8"))

    candidates = {category: [] for category in CATEGORY_FOLDERS}
    for item in tree.get("tree", []):
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        parts = path.split("/")
        if len(parts) < 2:
            continue

        category = parts[0]
        suffix = Path(path).suffix.lower()
        if category in candidates and suffix in IMAGE_EXTENSIONS:
            candidates[category].append(path)

    rng = random.Random(RANDOM_SEED)
    # Spread the sample across the available shirt-related folders.
    chosen = []
    remaining = TARGET
    active = [c for c in CATEGORY_FOLDERS if candidates[c]]

    while remaining and active:
        per_category = max(1, remaining // len(active))
        next_active = []
        for category in active:
            pool = candidates[category]
            take = min(per_category, len(pool))
            if take:
                chosen.extend(rng.sample(pool, take))
                remaining -= take
            if len(pool) > take:
                next_active.append(category)
        active = next_active

    if remaining:
        already = set(chosen)
        extra_pool = [
            path
            for category in CATEGORY_FOLDERS
            for path in candidates[category]
            if path not in already
        ]
        take = min(remaining, len(extra_pool))
        if take:
            chosen.extend(rng.sample(extra_pool, take))
            remaining -= take

    if remaining:
        raise RuntimeError(
            f"Only {TARGET - remaining} suitable images were found; "
            "the source repository may have changed."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_lines = [
        "EyeTheia shirt image sample",
        "Source: https://github.com/lightly-ai/dataset_clothing_images",
        "Repository license: CC0-1.0 (verify source license before reuse)",
        "",
    ]

    downloaded = 0
    for index, source_path in enumerate(chosen, start=1):
        suffix = Path(source_path).suffix.lower()
        destination = OUTPUT_DIR / f"shirt_{index:03d}{suffix}"
        raw_url = RAW_BASE + source_path

        try:
            data = fetch_bytes(raw_url)
            if len(data) < 1024:
                raise ValueError("Downloaded file is unexpectedly small.")
            destination.write_bytes(data)
            downloaded += 1
            manifest_lines.append(
                f"{destination.name}\t{source_path}"
            )
            print(f"[{downloaded:02d}/{TARGET}] {destination.name}")
        except Exception as exc:
            print(f"Skipping {source_path}: {exc}")
        time.sleep(0.05)

    if downloaded != TARGET:
        raise RuntimeError(
            f"Downloaded {downloaded}/{TARGET}. "
            "Delete the partial output folder and run again."
        )

    (OUTPUT_DIR / "source_manifest.txt").write_text(
        "\n".join(manifest_lines), encoding="utf-8"
    )

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(OUTPUT_DIR.iterdir()):
            archive.write(file_path, arcname=f"shirt_images_30/{file_path.name}")

    print("\nFinished.")
    print(f"Image folder: {OUTPUT_DIR}")
    print(f"ZIP archive:  {ZIP_PATH}")
    print("Copy the 30 image files into your EyeTheia/shopping_assets folder.")


if __name__ == "__main__":
    main()
