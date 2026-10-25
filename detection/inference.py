import os
import json
import logging
from typing import Dict, Any, Tuple, Optional, Union
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from .model import CoffeeDiseaseClassifier

logger = logging.getLogger(__name__)

# Exact class order from training checkpoint
CLASS_NAMES = ['Leaf Miner', 'Healthy', 'Phoma', 'Rust']

# Standard ImageNet normalization used during training
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_default_device() -> torch.device:
    """Returns CUDA if available and healthy, else CPU."""
    if torch.cuda.is_available():
        try:
            # Quick check if CUDA tensor creation works
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def resolve_model_path(custom_path: Optional[str] = None) -> Path:
    """Resolve model checkpoint path from multiple potential locations."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    if os.environ.get("COFFEE_MODEL_PATH"):
        candidates.append(Path(os.environ["COFFEE_MODEL_PATH"]))

    base_dir = Path(__file__).resolve().parent.parent
    candidates.extend([
        base_dir / "models" / "coffee_leaf_disease_model.pth",
        base_dir.parent / "Coffee-Leaf-Disease-Detector" / "coffee_leaf_disease_model.pth",
        Path("models/coffee_leaf_disease_model.pth"),
        Path("coffee_leaf_disease_model.pth")
    ])

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Coffee leaf disease model checkpoint not found. Checked: {[str(c) for c in candidates]}"
    )


def load_labels_metadata() -> Dict[str, Any]:
    """Load class descriptions and symptoms from labels.json."""
    labels_file = Path(__file__).resolve().parent / "labels.json"
    if labels_file.is_file():
        with open(labels_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"classes": CLASS_NAMES, "translations": {"en": {}, "am": {}}}


def load_vision_model(
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Tuple[Optional[CoffeeDiseaseClassifier], Optional[str]]:
    """
    Loads the trained CoffeeDiseaseClassifier ResNet18 model checkpoint.
    Returns (model, error_message).
    """
    dev = device or get_default_device()
    try:
        resolved_path = resolve_model_path(model_path)
        logger.info(f"Loading vision model from: {resolved_path} on {dev}")

        model = CoffeeDiseaseClassifier(num_classes=len(CLASS_NAMES))
        checkpoint = torch.load(resolved_path, map_location=dev)

        # Check if state_dict is nested
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Verify key compatibility
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        if missing_keys:
            logger.warning(f"Missing keys during checkpoint load: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys during checkpoint load: {unexpected_keys}")

        model.to(dev)
        model.eval()
        return model, None

    except FileNotFoundError as fnf:
        err = f"Model file error: {str(fnf)}"
        logger.error(err)
        return None, err
    except Exception as e:
        err = f"Failed to load vision model: {str(e)}"
        logger.error(err, exc_info=True)
        return None, err


def preprocess_image(image_input: Union[str, Path, Image.Image, bytes]) -> Tuple[Optional[torch.Tensor], Optional[Image.Image], Optional[str]]:
    """
    Validates, converts to RGB, and applies standard inference preprocessing.
    Returns (tensor_batch, pil_image, error_message).
    """
    try:
        pil_image: Optional[Image.Image] = None
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(str(image_input))
        elif isinstance(image_input, Image.Image):
            pil_image = image_input
        elif isinstance(image_input, (bytes, bytearray)):
            import io
            pil_image = Image.open(io.BytesIO(image_input))
        elif hasattr(image_input, "read"):
            # File-like object (e.g. Streamlit UploadedFile)
            pil_image = Image.open(image_input)
        else:
            return None, None, f"Unsupported image input type: {type(image_input)}"

        # Validate image format and dimensions
        if pil_image.width < 10 or pil_image.height < 10:
            return None, None, f"Image dimensions too small ({pil_image.width}x{pil_image.height}). Must be at least 10x10 pixels."

        # Ensure RGB format
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Apply transforms and add batch dimension [1, 3, 224, 224]
        tensor = INFERENCE_TRANSFORMS(pil_image).unsqueeze(0)
        return tensor, pil_image, None

    except Exception as e:
        return None, None, f"Invalid or unreadable image: {str(e)}"


def predict_leaf_disease(
    model: CoffeeDiseaseClassifier,
    image_input: Union[str, Path, Image.Image, Any],
    device: Optional[torch.device] = None,
    lang: str = "en"
) -> Dict[str, Any]:
    """
    Runs leaf disease classification inference.
    Returns a comprehensive prediction result dictionary.
    """
    if model is None:
        return {
            "success": False,
            "error": "Model is not loaded or initialized.",
            "predicted_class": None,
            "display_name": None,
            "confidence": 0.0,
            "probabilities": {},
            "metadata": {}
        }

    dev = device or get_default_device()

    # Preprocess image
    tensor, pil_img, err = preprocess_image(image_input)
    if err is not None:
        return {
            "success": False,
            "error": err,
            "predicted_class": None,
            "display_name": None,
            "confidence": 0.0,
            "probabilities": {},
            "metadata": {}
        }

    labels_data = load_labels_metadata()
    lang_key = "am" if lang == "am" else "en"
    translations = labels_data.get("translations", {}).get(lang_key, {})

    try:
        tensor = tensor.to(dev)
        # Use torch.inference_mode for memory efficiency and faster execution
        with torch.inference_mode():
            logits = model(tensor)
            probabilities = F.softmax(logits[0], dim=0)

        probs_list = [p.item() for p in probabilities]
        max_idx = int(torch.argmax(probabilities).item())
        predicted_class = CLASS_NAMES[max_idx]
        confidence = probs_list[max_idx]

        prob_dict = {
            CLASS_NAMES[i]: round(probs_list[i] * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        class_meta = translations.get(predicted_class, {})
        display_name = class_meta.get("display_name", predicted_class)

        return {
            "success": True,
            "error": None,
            "predicted_class": predicted_class,
            "display_name": display_name,
            "confidence": round(confidence * 100, 2),
            "raw_confidence": confidence,
            "probabilities": prob_dict,
            "pil_image": pil_img,
            "metadata": class_meta,
            "all_translations": labels_data.get("translations", {})
        }

    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Inference execution failed: {str(e)}",
            "predicted_class": None,
            "display_name": None,
            "confidence": 0.0,
            "probabilities": {},
            "metadata": {}
        }
