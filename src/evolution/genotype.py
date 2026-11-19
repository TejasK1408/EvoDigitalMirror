"""Continuous genotype encoding for the evolutionary fashion system."""

# Each feature has a fixed, ordered set of possible categories.
FEATURE_SCHEMA = {
    "color": ["#FFFFFF", "#000000", "#FF0000", "#008000", "#0000FF","#FFFF00", "#FFA500",
               "#800080", "#00FFFF", "#FFC0CB","#808080", "#A52A2A", "#800000", "#008080", "#FFD700", "#C0C0C0", "#00FF00",
                 "#FF00FF", "#F5F5DC", "#D3D3D3","#B22222", "#FF4500", "#2E8B57", "#DAA520", "#4B0082", "#7FFF00", "#FF1493",
                   "#00CED1", "#8B0000", "#FF6347",],
    "sleeve_length": ["sleeveless",
    "cap_sleeve",
    "short_sleeve",
    "elbow_length",
    "three_quarter",
    "long_sleeve",
    "asymmetric",
    "not_applicable"],
    "neckline":["crew_round",
    "v_neck",
    "scoop_neck",
    "square_neck",
    "boat_neck",
    "sweetheart",
    "turtleneck_mock",
    "collared_shirt",
    "off_shoulder",
    "halter",
    "hooded",
    "not_applicable"],
    "length": ["cropped",
    "waist_length",
    "hip_length",
    "mini_short",
    "knee_length",
    "midi",
    "maxi_full_length",
    "high_low"],
    "silhouette": ["bodycon_tight",
    "slim_tailored",
    "regular_straight",
    "relaxed_oversized",
    "a_line",
    "fit_and_flare",
    "empire_waist",
    "wrap",
    "boxy"],
    "pattern": ["solid_plain",
    "color_block",
    "striped",
    "plaid_checkered",
    "polka_dot",
    "floral",
    "geometric_abstract",
    "animal_print",
    "graphic_logo",
    "tie_dye"],
}


def generate_category_values(number_of_options):
    """Return equally spaced values strictly between 0 and 1."""
    if number_of_options < 1:
        raise ValueError("A feature must have at least one category.")
    return [(k + 1) / (number_of_options + 1) for k in range(number_of_options)]


# Build category -> continuous gene mappings from FEATURE_SCHEMA.
CATEGORY_MAPPING = {
    feature: dict(zip(categories, generate_category_values(len(categories))))
    for feature, categories in FEATURE_SCHEMA.items()
}


# IMPORTANT:
# These six attribute records are DEMONSTRATION VALUES so that the integration
# can run immediately. Replace each record with the characteristics visible
# in the corresponding shopping_assets/dressN.jfif image.
# Do not change the feature names unless you also update FEATURE_SCHEMA.
DRESS_ATTRIBUTES = {
    "ITEMS": {
        1: {"color": "#000000", "sleeve_length": "short_sleeve", "neckline": "crew_round", "length": "midi", "silhouette": "a_line", "pattern": "solid_plain"},
        2: {"color": "#FFFFFF", "sleeve_length": "sleeveless", "neckline": "v_neck", "length": "maxi_full_length", "silhouette": "regular_straight", "pattern": "floral"},
        3: {"color": "#FF0000", "sleeve_length": "long_sleeve", "neckline": "square_neck", "length": "knee_length", "silhouette": "fit_and_flare", "pattern": "graphic_logo"},
        4: {"color": "#0000FF", "sleeve_length": "three_quarter", "neckline": "collared_shirt", "length": "mini_short", "silhouette": "bodycon_tight", "pattern": "striped"},
        5: {"color": "#008000", "sleeve_length": "short_sleeve", "neckline": "sweetheart", "length": "midi", "silhouette": "empire_waist", "pattern": "geometric_abstract"},
        6: {"color": "#FFFF00", "sleeve_length": "sleeveless", "neckline": "crew_round", "length": "maxi_full_length", "silhouette": "a_line", "pattern": "floral"},
        7: {"color": "#FFA500", "sleeve_length": "long_sleeve", "neckline": "square_neck", "length": "knee_length", "silhouette": "fit_and_flare", "pattern": "graphic_logo"},
        8: {"color": "#800080", "sleeve_length": "short_sleeve", "neckline": "sweetheart", "length": "midi", "silhouette": "empire_waist", "pattern": "geometric_abstract"},
        9: {"color": "#00FFFF", "sleeve_length": "sleeveless", "neckline": "halter", "length": "maxi_full_length", "silhouette": "wrap", "pattern": "tie_dye"},
        10: {"color": "#FF00FF", "sleeve_length": "long_sleeve", "neckline": "off_shoulder", "length": "knee_length", "silhouette": "fit_and_flare", "pattern": "floral"},
        11: {"color": "#808080", "sleeve_length": "three_quarter", "neckline": "collared_shirt", "length": "mini_short", "silhouette": "bodycon_tight", "pattern": "striped"},
        12: {"color": "#A52A2A", "sleeve_length": "short_sleeve", "neckline": "v_neck", "length": "midi", "silhouette": "regular_straight", "pattern": "solid_plain"},
        13: {"color": "#800000", "sleeve_length": "sleeveless", "neckline": "crew_round", "length": "maxi_full_length", "silhouette": "a_line", "pattern": "floral"},
        14: {"color": "#008080", "sleeve_length": "long_sleeve", "neckline": "square_neck", "length": "knee_length", "silhouette": "fit_and_flare", "pattern": "graphic_logo"},
        15: {"color": "#FFD700", "sleeve_length": "short_sleeve", "neckline": "sweetheart", "length": "midi", "silhouette": "empire_waist", "pattern": "geometric_abstract"},
        16: {"color": "#C0C0C0", "sleeve_length": "sleeveless", "neckline": "halter", "length": "maxi_full_length", "silhouette": "wrap", "pattern": "tie_dye"},
        17: {"color": "#00FF00", "sleeve_length": "long_sleeve", "neckline": "off_shoulder", "length": "knee_length", "silhouette": "fit_and_flare", "pattern": "floral"},
        18: {"color": "#F5F5DC", "sleeve_length": "three_quarter", "neckline": "collared_shirt", "length": "mini_short", "silhouette": "bodycon_tight", "pattern": "striped"},
        19: {"color": "#D3D3D3", "sleeve_length": "short_sleeve", "neckline": "v_neck", "length": "midi", "silhouette": "regular_straight", "pattern": "solid_plain"},
        20: {"color": "#B22222", "sleeve_length": "sleeveless", "neckline": "crew_round", "length": "maxi_full_length", "silhouette": "a_line", "pattern": "floral"},
    }

}

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

def encode_dress(dress):
    """Encode one attribute dictionary as a continuous genotype list."""
    missing = [feature for feature in FEATURE_SCHEMA if feature not in dress]
    if missing:
        raise ValueError(f"Missing dress features: {', '.join(missing)}")

    genotype = []
    for feature, categories in FEATURE_SCHEMA.items():
        category = dress[feature]
        if category not in CATEGORY_MAPPING[feature]:
            raise ValueError(
                f"Unknown category {category!r} for {feature!r}. "
                f"Choose one of: {categories}"
            )
        genotype.append(CATEGORY_MAPPING[feature][category])
    return genotype


def encode_population(dresses):
    """Encode a sequence of dress attribute dictionaries."""
    return [encode_dress(dress) for dress in dresses]


def display_genotype(dress_number, genotype):
    """Print a labelled genotype to the terminal."""
    print(f"\nDress {dress_number} genotype:")
    for feature, gene in zip(FEATURE_SCHEMA, genotype):
        print(f"  {feature:12s}: {gene:.4f}")

def decode_gene(feature, gene):
    """Snap one continuous gene to the nearest category of its feature."""
    categories = FEATURE_SCHEMA[feature]
    values = generate_category_values(len(categories))
    best = min(range(len(categories)), key=lambda i: abs(values[i] - gene))
    return categories[best]

def decode_genotype(genotype):
    """Convert a six-gene list back into a {feature: category} dict."""
    return {f: decode_gene(f, g) for f, g in zip(FEATURE_SCHEMA, genotype)}