"""
Continuous-genotype dress generator prototype.

The six genes are used as continuous controls; this file does NOT snap them
to the nearest category. The prompt-based conditioning is a proof of concept,
not a trained genotype-to-image model.

Run from the EyeTheia project root:
    python src/evolution/image_generator.py

Or provide a genotype:
    python src/evolution/image_generator.py --genotype 0.43 0.62 0.28 0.71 0.55 0.36

Dependencies:
    pip install torch diffusers transformers accelerate safetensors peft pillow
"""

from __future__ import annotations

import argparse
import colorsys
import random
from pathlib import Path
from typing import Sequence

import os
import logging
import time

from matplotlib.pylab import positive

from dotenv import load_dotenv
from google import genai
from PIL import Image

try:
    from evolution.genotype import decode_genotype, ATTRIBUTE_DESCRIPTIONS
except ImportError:
    from .genotype import decode_genotype, ATTRIBUTE_DESCRIPTIONS




load_dotenv()

MODEL_ID = "gemini-3.1-flash-image"

_CLIENT = None

logger = logging.getLogger(__name__)




def load_client():
    """Initialize and reuse the Gemini API client."""
    global _CLIENT

    if _CLIENT is not None:
        return _CLIENT

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Check your .env file."
        )

    _CLIENT = genai.Client(api_key=api_key)

    logger.info("Gemini image client initialized | model=%s", MODEL_ID)

    return _CLIENT



PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "shopping_assets" / "offspring_dress.png"




def validate_genotype(genotype: Sequence[float]) -> list[float]:
    """Validate and return exactly six continuous genes in the open interval (0, 1)."""
    if len(genotype) != 6:
        raise ValueError(
            f"Expected exactly 6 genes (color, sleeve, neckline, length, "
            f"silhouette, pattern); received {len(genotype)}."
        )

    values = [float(value) for value in genotype]
    for index, value in enumerate(values, start=1):
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"Gene {index} must be strictly between 0 and 1; got {value}."
            )
    return values




def genotype_to_prompt(genotype: Sequence[float]) -> tuple[str, str]:
                """Decode the six genes into categories and describe them in text."""
                values = validate_genotype(genotype)
                decoded = decode_genotype(values)

                spec = "\n".join(
                    f"- {feature}: {ATTRIBUTE_DESCRIPTIONS[feature][category]}"
                    for feature, category in decoded.items()
                )

                positive = f"""Design one complete women's fashion dress.

                GARMENT SPECIFICATION:
                {spec}

                VISUAL REQUIREMENTS:
                Show the complete dress from the front, uncropped.
                Use realistic fabric texture and natural garment construction.
                Use a clean, light-grey studio background and even lighting.
                Do not add text, watermarks, logos, or extra garments."""

                return positive, ""





def generate_offspring_image(
    genotype: Sequence[float],
    output_path: str | Path = DEFAULT_OUTPUT,
    steps: int = 6,           # Retained for caller compatibility
    guidance_scale: float = 1.5,  # Retained for caller compatibility
    seed: int | None = None,
    parent_images: Sequence[str | Path] | None = None,
) -> Path:

    values = validate_genotype(genotype)
    prompt, _ = genotype_to_prompt(values)

    client = load_client()

    # Add parent images as visual references when provided.
    contents = [prompt]
    if parent_images:
        contents.append("The following images are parent garment references. "
            "Use them as inspiration, but create a new design that "
            "follows the specified genotype.")
        for image_path in parent_images:
            image_path = Path(image_path)
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            contents.append(Image.open(image_path).convert("RGB"))

    logger.info(
        "Generating offspring | model=%s | seed=%s | references=%s",
        MODEL_ID, seed, len(parent_images or [])
    )

    response = None
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
            )
            break
        except Exception as e:
            logger.warning("Generation attempt %d failed: %s", attempt, e)
            if attempt == 3:
                raise
            time.sleep(2 * attempt)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    image_saved = False

    for part in (response.parts or []):
        if part.inline_data is not None:
            generated_image = part.as_image()
            generated_image.save(destination)
            image_saved = True
            break

    if not image_saved:
        raise RuntimeError(
            "Gemini returned no image. Check the API response and model."
        )

    logger.info("Offspring image saved | path=%s", destination)

    return destination


def main():
    parser = argparse.ArgumentParser(
        description="Generate a dress from six continuous genotype values."
    )
    parser.add_argument(
        "--genotype",
        nargs=6,
        type=float,
        default=[0.43, 0.62, 0.28, 0.71, 0.55, 0.36],
        metavar=("COLOR", "SLEEVE", "NECKLINE", "LENGTH", "FIT", "PATTERN"),
        help="Six values strictly between 0 and 1.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output image path.",
    )
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    generate_offspring_image(
        genotype=args.genotype,
        output_path=args.output,
        steps=args.steps,
        seed=args.seed,
        parent_images=[
            parent for parent in [
                PROJECT_ROOT / "shopping_assets" / "dress1.jfif",
                PROJECT_ROOT / "shopping_assets" / "dress2.jfif"
            ] if parent.is_file()
        ]
    )


if __name__ == "__main__":
    main()
