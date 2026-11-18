
import torch
from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel


# --------------------------------------------------
# 1. Import the existing genotype definitions
# --------------------------------------------------

try:
    from evolution.genotype import FEATURE_SCHEMA, encode_dress
    from evolution.genotype import ATTRIBUTE_DESCRIPTIONS  # inside the existing try/except import
except ImportError:
    from .genotype import FEATURE_SCHEMA, encode_dress


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

ATTRIBUTE_DESCRIPTIONS = {
    "color": {
        "#FFFFFF": "white",
        "#000000": "black",
        "#FF0000": "red",
        "#008000": "green",
        "#0000FF": "blue",
        "#FFFF00": "yellow",
        "#FFA500": "orange",
        "#800080": "purple",
        "#00FFFF": "cyan",
        "#FFC0CB": "pink",
        "#808080": "gray",
        "#A52A2A": "brown",
        "#800000": "maroon",
        "#008080": "teal",
        "#FFD700": "gold",
        "#C0C0C0": "silver",
        "#00FF00": "lime green",
        "#FF00FF": "magenta",
        "#F5F5DC": "beige",
        "#D3D3D3": "light gray",
        "#B22222": "firebrick red",
        "#FF4500": "orange red",
        "#2E8B57": "sea green",
        "#DAA520": "goldenrod yellow",
        "#4B0082": "indigo",
        "#7FFF00": "chartreuse green",
        "#FF1493": "deep pink",
        "#00CED1": "dark turquoise",
        "#8B0000": "dark red",
        "#FF6347": "tomato red",
    },

    "sleeve_length": {
        "sleeveless": "sleeveless",
        "cap_sleeve": "cap sleeves",
        "short_sleeve": "short sleeves",
        "elbow_length": "elbow-length sleeves",
        "three_quarter": "three-quarter sleeves",
        "long_sleeve": "long sleeves",
        "asymmetric": "asymmetric single-shoulder sleeve",
        "not_applicable": "garment without sleeves or upper bodice",
    },

    "neckline": {
        "crew_round": "round crew neckline",
        "v_neck": "V-shaped neckline",
        "scoop_neck": "scoop neckline",
        "square_neck": "square neckline",
        "boat_neck": "wide boat neckline",
        "sweetheart": "sweetheart neckline",
        "turtleneck_mock": "high turtleneck or mock neckline",
        "collared_shirt": "collared shirt neckline",
        "off_shoulder": "off-the-shoulder neckline",
        "halter": "halter neckline",
        "hooded": "hooded neckline",
        "not_applicable": "garment without a neckline",
    },

    "length": {
        "cropped": "cropped length above the waist",
        "waist_length": "waist-length",
        "hip_length": "hip-length",
        "mini_short": "mini short length",
        "knee_length": "knee-length",
        "midi": "mid-calf midi length",
        "maxi_full_length": "full-length maxi",
        "high_low": "asymmetric high-low hemline",
    },

    "silhouette": {
        "bodycon_tight": "tight form-fitting bodycon silhouette",
        "slim_tailored": "slim tailored silhouette",
        "regular_straight": "regular straight silhouette",
        "relaxed_oversized": "relaxed oversized silhouette",
        "a_line": "flared A-line silhouette",
        "fit_and_flare": "fit and flare silhouette",
        "empire_waist": "high empire waist silhouette",
        "wrap": "wrap-style silhouette",
        "boxy": "loose boxy silhouette",
    },

    "pattern": {
        "solid_plain": "plain solid single-color design",
        "color_block": "color block pattern",
        "striped": "striped pattern",
        "plaid_checkered": "plaid or checkered pattern",
        "polka_dot": "polka dot pattern",
        "floral": "floral print pattern",
        "geometric_abstract": "geometric or abstract pattern",
        "animal_print": "animal print pattern",
        "graphic_logo": "graphic print or logo design",
        "tie_dye": "tie-dye pattern",
    },
}



def make_prompt(attribute, category):
    """
    Create a natural-language description
    for one possible dress attribute.
    """

    description = ATTRIBUTE_DESCRIPTIONS[attribute][category]

    return f"A photograph of a women's dress with {description}."


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
