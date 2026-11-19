
import torch
from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel


# --------------------------------------------------
# 1. Import the existing genotype definitions
# --------------------------------------------------

try:
    from evolution.genotype import FEATURE_SCHEMA, ATTRIBUTE_DESCRIPTIONS, encode_dress
except ImportError:
    from .genotype import FEATURE_SCHEMA, ATTRIBUTE_DESCRIPTIONS, encode_dress


# --------------------------------------------------
# 2. CLIP model configuration
# --------------------------------------------------

MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_processor = None


def load_clip_model():
    """
    Load CLIP only once.
    Subsequent image encodings reuse the same model.
    """
    global _model, _processor

    if _model is None:
        print("Loading CLIP image encoder...")

        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)

        _model = CLIPModel.from_pretrained(MODEL_NAME)
        _model.to(DEVICE)
        _model.eval()

        print(f"CLIP loaded successfully on {DEVICE}")

    return _model, _processor


# --------------------------------------------------
# 3. Convert genotype categories into readable text
# --------------------------------------------------





def make_prompt(attribute, category):
    d = ATTRIBUTE_DESCRIPTIONS[attribute][category]
    if attribute == "color":
        return f"A photograph of a {d} women's dress."
    return f"A photograph of a women's dress with {d}."


# --------------------------------------------------
# 4. Predict one attribute from an image
# --------------------------------------------------

def predict_attribute(image, attribute, model, processor):
    """
    Compare one dress image against all possible
    categories for a particular attribute.
    """

    categories = FEATURE_SCHEMA[attribute]

    prompts = [
        make_prompt(attribute, category)
        for category in categories
    ]

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

        scores = outputs.logits_per_image[0]

        probabilities = torch.softmax(scores, dim=0)

    best_index = torch.argmax(probabilities).item()

    predicted_category = categories[best_index]

    confidence = probabilities[best_index].item()

    return predicted_category, confidence


# --------------------------------------------------
# 5. Encode one dress image
# --------------------------------------------------

def encode_image(image_path):
    """
    Input:
        Path to a dress image.

    Output:
        Detected attributes and numerical genotype.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Dress image not found: {image_path}"
        )

    model, processor = load_clip_model()

    image = Image.open(image_path).convert("RGB")

    attributes = {}
    confidence_scores = {}

    print(f"\nAnalysing: {image_path.name}")

    # Predict all six attributes
    for attribute in FEATURE_SCHEMA:

        category, confidence = predict_attribute(
            image,
            attribute,
            model,
            processor
        )

        attributes[attribute] = category
        confidence_scores[attribute] = round(confidence, 4)

        print(
            f"{attribute:12s}: {category:15s} "
            f"(confidence: {confidence:.3f})"
        )

    # Convert detected attributes into numerical genes
    genotype = encode_dress(attributes)

    print("\nGenerated genotype:")
    print(genotype)

    return {
        "image_path": str(image_path),
        "attributes": attributes,
        "confidence": confidence_scores,
        "genotype": genotype
    }


# --------------------------------------------------
# 6. Encode both selected parent images
# --------------------------------------------------

def encode_selected_images(image_paths):
    """
    Input:
        List containing the two selected image paths.

    Output:
        Encoded information for both parent dresses.
    """

    if len(image_paths) != 2:
        raise ValueError(
            "Exactly two selected parent images are required."
        )

    parent_results = []

    for image_path in image_paths:

        result = encode_image(image_path)

        parent_results.append(result)

    print("\n========== PARENT GENOTYPES ==========")

    for index, result in enumerate(parent_results, start=1):

        print(f"\nParent {index}")
        print("Image:", result["image_path"])
        print("Genotype:", result["genotype"])

    return parent_results
