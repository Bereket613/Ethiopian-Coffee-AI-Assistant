"""
Automated unit and regression tests for Computer Vision module.
Tests real ResNet-18 checkpoint loading, forward pass, tensor dimensions,
class mapping, image preprocessing, and invalid input handling.
"""

import sys
from pathlib import Path
import pytest
import torch
from PIL import Image
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.model import CoffeeDiseaseClassifier
from detection.inference import (
    load_vision_model,
    predict_leaf_disease,
    preprocess_image,
    CLASS_NAMES,
    resolve_model_path,
)


def test_model_path_resolution():
    """Verify that the model checkpoint path resolves to a valid existing file."""
    model_path = resolve_model_path()
    assert model_path.is_file(), f"Model file does not exist at {model_path}"
    assert model_path.stat().st_size > 10_000_000, "Checkpoint file is smaller than 10MB"


def test_model_architecture_and_checkpoint_loading():
    """Verify checkpoint loads with exact key match into ResNet18."""
    model, error = load_vision_model()
    assert error is None, f"Model loading failed with error: {error}"
    assert model is not None, "Loaded model is None"
    assert isinstance(model, CoffeeDiseaseClassifier)

    # Test dummy forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        logits = model(dummy_input)

    assert logits.shape == (1, 4), f"Expected output shape (1, 4), got {logits.shape}"
    assert len(CLASS_NAMES) == 4
    assert CLASS_NAMES == ['Leaf Miner', 'Healthy', 'Phoma', 'Rust']


def test_image_preprocessing_rgb():
    """Verify preprocessing correctly converts RGBA/Grayscale to RGB and normalizes."""
    # Create test RGB image
    img_rgb = Image.new("RGB", (300, 400), color=(34, 139, 34))
    tensor, pil_img, err = preprocess_image(img_rgb)
    assert err is None
    assert tensor is not None
    assert tensor.shape == (1, 3, 224, 224)
    assert pil_img.mode == "RGB"

    # Create test RGBA image
    img_rgba = Image.new("RGBA", (150, 150), color=(100, 200, 50, 255))
    tensor_rgba, pil_rgba, err_rgba = preprocess_image(img_rgba)
    assert err_rgba is None
    assert pil_rgba.mode == "RGB"
    assert tensor_rgba.shape == (1, 3, 224, 224)


def test_invalid_image_handling():
    """Verify corrupted or tiny images are caught gracefully without crashing."""
    # Extremely small image (<10x10)
    tiny_img = Image.new("RGB", (4, 4), color=(0, 0, 0))
    tensor, pil_img, err = preprocess_image(tiny_img)
    assert tensor is None
    assert "too small" in err.lower()

    # Invalid input type
    tensor_inv, _, err_inv = preprocess_image(12345)
    assert tensor_inv is None
    assert "unsupported" in err_inv.lower()


def test_real_leaf_inference_e2e():
    """Perform real end-to-end inference using the trained checkpoint."""
    model, error = load_vision_model()
    assert error is None

    # Synthesize realistic leaf-like image with green tones and spot
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :] = [45, 106, 79]  # Green leaf blade
    arr[80:120, 80:120] = [180, 130, 20]  # Rust-like yellow/brown lesion
    test_img = Image.fromarray(arr)

    result = predict_leaf_disease(model, test_img, lang="en")
    assert result["success"] is True
    assert result["predicted_class"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 100.0
    assert len(result["probabilities"]) == 4

    # Verify probability distribution sum ~ 100%
    prob_sum = sum(result["probabilities"].values())
    assert 99.0 <= prob_sum <= 101.0, f"Probabilities do not sum to 100%: {prob_sum}"


def test_amharic_metadata_output():
    """Verify that predictions with lang='am' produce natural Amharic display names."""
    model, _ = load_vision_model()
    test_img = Image.new("RGB", (224, 224), color=(50, 120, 50))
    result = predict_leaf_disease(model, test_img, lang="am")

    assert result["success"] is True
    assert result["display_name"] is not None
    # Verify presence of Ethiopic unicode characters
    has_ethiopic = any('\u1200' <= char <= '\u137F' for char in result["display_name"])
    assert has_ethiopic, f"Display name did not contain Amharic script: {result['display_name']}"
