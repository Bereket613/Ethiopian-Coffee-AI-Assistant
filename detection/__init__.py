"""
Computer Vision detection module for Ethiopian Coffee AI Assistant.
"""

from .model import CoffeeDiseaseClassifier
from .inference import (
    load_vision_model,
    predict_leaf_disease,
    CLASS_NAMES,
    preprocess_image,
)

__all__ = [
    "CoffeeDiseaseClassifier",
    "load_vision_model",
    "predict_leaf_disease",
    "CLASS_NAMES",
    "preprocess_image",
]
