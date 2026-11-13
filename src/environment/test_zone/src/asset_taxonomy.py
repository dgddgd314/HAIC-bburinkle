from __future__ import annotations

import re


ASSET_BROWSE_CATEGORIES = (
    "Seating",
    "Tables & Desks",
    "Storage",
    "Beds & Bedding",
    "Kitchen & Dining",
    "Electronics & Appliances",
    "Office & School",
    "Bathroom",
    "Lighting",
    "Decor & Art",
    "Plants & Nature",
    "Lab & Tools",
    "Sports & Recreation",
    "Toys",
    "Architecture",
    "Containers & Misc.",
    "Other",
)


ASSET_CATEGORY_KEYWORDS = {
    "Seating": (
        "chair", "armchair", "sofa", "couch", "stool", "bench", "seat",
    ),
    "Tables & Desks": (
        "table", "desk", "counter", "workbench", "nightstand", "side table",
    ),
    "Storage": (
        "cabinet", "shelf", "shelving", "bookcase", "dresser", "wardrobe",
        "drawer", "locker", "rack", "cupboard",
    ),
    "Beds & Bedding": (
        "bed", "mattress", "pillow", "blanket", "bedsheet", "duvet",
    ),
    "Kitchen & Dining": (
        "kitchen", "plate", "bowl", "cup", "mug", "glass", "fork", "spoon",
        "knife", "pot", "pan", "food", "fruit", "apple", "banana", "bread",
        "bottle", "refrigerator", "fridge", "microwave", "toaster", "oven",
        "stove", "dishwasher", "coffee machine",
    ),
    "Electronics & Appliances": (
        "laptop", "computer", "monitor", "television", "cellphone", "phone",
        "tablet computer", "keyboard", "mouse", "printer", "camera", "speaker",
        "projector", "washing machine", "vacuum", "fan", "electronics",
    ),
    "Office & School": (
        "book", "notebook", "pen", "pencil", "whiteboard", "blackboard",
        "chalkboard", "school", "classroom", "office", "folder", "paper",
    ),
    "Bathroom": (
        "toilet", "bathtub", "bath tub", "shower", "bathroom", "towel",
        "toothbrush", "soap", "washbasin", "bath sink",
    ),
    "Lighting": (
        "lamp", "light fixture", "chandelier", "lantern", "ceiling light",
    ),
    "Decor & Art": (
        "painting", "poster", "wall decor", "artwork", "sculpture", "statue",
        "bust", "figurine", "vase", "clock", "mirror", "picture frame", "rug",
        "carpet", "curtain",
        "decoration", "decorative",
    ),
    "Plants & Nature": (
        "plant", "flower", "tree", "grass", "shrub", "bush", "foliage",
    ),
    "Lab & Tools": (
        "microscope", "beaker", "flask", "laboratory", "lab equipment", "tool",
        "hammer", "drill", "grinder", "wrench", "screwdriver", "machine", "equipment",
    ),
    "Sports & Recreation": (
        "sports", "ball", "basketball", "football", "soccer", "tennis", "racket",
        "racquet", "exercise", "gym", "bicycle", "bike", "skateboard",
    ),
    "Toys": (
        "toy", "teddy", "doll", "plush", "game", "puzzle", "blocks", "character",
    ),
    "Architecture": (
        "doorway", "door", "window", "stair", "column", "railing", "partition",
        "room divider", "fireplace",
    ),
    "Containers & Misc.": (
        "container", "basket", "box", "bin", "trash can", "bucket", "case",
        "cart", "trolley",
    ),
}

ASSET_CATEGORY_PATTERNS = {
    category: tuple(
        re.compile(rf"\b{re.escape(keyword)}(?:s|es)?\b")
        for keyword in keywords
    )
    for category, keywords in ASSET_CATEGORY_KEYWORDS.items()
}


def _normalized_text(values) -> str:
    text = " ".join(str(value) for value in values if value)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text).lower()
    text = re.sub(r"[_/\\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _category_from_text(text: str) -> str | None:
    match_priority = (
        "Lab & Tools", "Toys", "Sports & Recreation", "Bathroom",
        "Plants & Nature", "Office & School", "Kitchen & Dining",
        "Electronics & Appliances", "Lighting", "Architecture", "Decor & Art",
        "Beds & Bedding", "Storage", "Tables & Desks", "Seating",
        "Containers & Misc.",
    )
    for category in match_priority:
        for pattern in ASSET_CATEGORY_PATTERNS[category]:
            if pattern.search(text):
                return category
    return None


def asset_browse_category(asset: dict) -> str:
    category = _category_from_text(
        _normalized_text(asset.get("categories", []))
    )
    if category is not None:
        return category
    fields = [
        asset.get("display_name", ""),
        asset.get("asset_id", ""),
        asset.get("path_category", ""),
        *asset.get("aliases", [])[:3],
    ]
    return _category_from_text(_normalized_text(fields)) or "Other"


def asset_default_dynamic(asset: dict) -> bool:
    text = _normalized_text(
        [
            asset.get("display_name", ""),
            asset.get("asset_id", ""),
            *asset.get("aliases", [])[:5],
            *asset.get("categories", [])[:3],
        ]
    )
    if re.search(r"\b(goal|goalpost|goal post|hoop|rack|racket|racquet)\b", text):
        return False
    return bool(
        re.search(
            r"\b(ball|soccerball|football|basketball|volleyball|baseball|softball)\b",
            text,
        )
    )
