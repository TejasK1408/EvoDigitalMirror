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

from dotenv import load_dotenv
from google import genai
from PIL import Image

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

_PIPE = None


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


def _continuous_color_description(hue_gene: float) -> str:
    """
    Convert gene 1 to a continuously varying hue.
    This does not choose the nearest entry from a fixed color-category list.
    """
    hue = hue_gene % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.88)
    rgb = tuple(round(channel * 255) for channel in (red, green, blue))
    hue_degrees = hue * 360.0
    return (
        f"a vivid, saturated garment color at hue {hue_degrees:.1f} degrees "
        f"on the color wheel (RGB {rgb[0]}, {rgb[1]}, {rgb[2]})"
    )


def genotype_to_prompt(genotype: Sequence[float]) -> tuple[str, str]:
    """
    Turn the six continuous genes into a human-readable conditioning prompt.

    Gene order:
      1. color hue
      2. sleeve coverage: 0 = sleeveless, 1 = full sleeves
      3. neckline depth: 0 = high/closed, 1 = deep/open
      4. garment length: 0 = mini, 1 = floor length
      5. fittedness: 0 = loose, 1 = body-hugging
      6. pattern density: 0 = plain, 1 = dense pattern
    """
    color, sleeve, neckline, length, fittedness, pattern = validate_genotype(genotype)

    sleeve_pct = sleeve * 100
    neckline_pct = neckline * 100
    length_pct = length * 100
    fitted_pct = fittedness * 100
    pattern_pct = pattern * 100

    # Keep every control continuous: no nearest-category selection or bins.
    sleeve_text = (
        f"sleeve coverage set to {sleeve_pct:.1f}% "
        "(0% means sleeveless and 100% means full-length sleeves)"
    )
    neckline_text = (
        f"neckline openness set to {neckline_pct:.1f}% "
        "(0% means a high closed neckline and 100% means a very deep open neckline)"
    )
    length_text = (
        f"garment length set to {length_pct:.1f}% along a continuous "
        "mini-length to floor-length scale"
    )
    fit_text = (
        f"silhouette fittedness set to {fitted_pct:.1f}% "
        "(0% means very loose and 100% means body-hugging)"
    )
    pattern_text = (
        f"pattern density set to {pattern_pct:.1f}% "
        "(0% means plain fabric and 100% means dense pattern coverage)"
    )

    positive = f"""
                    Design one complete women's fashion dress.

                    GARMENT SPECIFICATION:
                    - Color: {_continuous_color_description(color)}
                    - Sleeve coverage: {sleeve_pct:.1f}% 
                    - Neckline openness: {neckline_pct:.1f}%
                    - Length: {length_pct:.1f}% from mini to floor length
                    - Fittedness: {fitted_pct:.1f}% from loose to body-hugging
                    - Pattern density: {pattern_pct:.1f}%

                    VISUAL REQUIREMENTS:
                    Create a physically coherent, professionally tailored garment.
                    Preserve the specified garment attributes.
                    Show the complete dress from the front.
                    Use realistic fabric texture and natural garment construction.
                    Use a clean, light-grey studio background and even lighting.

                    Do not add text, watermarks, logos, or extra garments.
                    Do not crop the dress.
                    """
    

    negative = ''
    return positive, negative


def load_pipeline():
    """Load the diffusion pipeline once and reuse it."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.load_lora_weights(LCM_LORA_ID)
    pipe.fuse_lora()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    if use_cuda:
        # Sequential offload is slower but more suitable for a 4 GB GPU.
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cpu")

    _PIPE = pipe
    return _PIPE



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
    contents = [
        prompt,
        (
            "The following images are parent garment references. "
            "Use them as visual inspiration, but create a new offspring "
            "design that follows the specified child genotype. "
            "Do not simply copy either parent."
        )
    ]

    if parent_images:
        for image_path in parent_images:
            image_path = Path(image_path)

            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            contents.append(Image.open(image_path).convert("RGB"))

    logger.info(
        "Generating offspring | model=%s | seed=%s | references=%s",
        MODEL_ID, seed, len(parent_images or [])
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=contents,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    image_saved = False

    for part in response.parts:
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
