"""
Shared domain constants for the language-spatial sensor project.

These define which NYU40 object categories and spatial relations are
considered valid for training and evaluation.  Both the data pipeline
(splits / filtering) and visualisation (BEV colouring) import from here
so there is a single source of truth.
"""

VALID_REGION_LABELS: frozenset[str] = frozenset({
    "bedroom",
    "office",
    "kitchen",
    "living room",
    "bathroom",
    "hall/stairwell",
    "garage",
    "rec room",
})

VALID_RELATIONS: frozenset[str] = frozenset({
    "on",
    "in",
    "near",
    "above",
    "below",
    "between",
})

# NYU40 label strings that are considered semantically meaningful for
# referential grounding.  Corresponds to the keys in viz/bev.py's _PALETTE.
VALID_NYU40_LABELS: frozenset[str] = frozenset({
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floormat",
    "clothes",
    "books",
    "refrigerator",
    "television",
    "paper",
    "towel",
    "showercurtain",
    "box",
    "whiteboard",
    "person",
    "nightstand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
    # "otherprop",
    # "otherfurniture",
    # "otherstructure",
})
