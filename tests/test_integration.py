"""
Automated End-to-End Integration Tests.
Validates the full workflow combining Vision, RAG Chatbot, and Amharic Support.
"""

import sys
from pathlib import Path
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.inference import load_vision_model, predict_leaf_disease
from chatbot.rag_chatbot import CoffeeRAGChatbot
from chatbot.translation import translation_service


def test_end_to_end_vision_to_chatbot_flow():
    """
    Test the critical integration requirement:
    1. A leaf image is classified by Computer Vision
    2. The prediction is stored in context
    3. The context is passed to the RAG Chatbot
    4. The chatbot retrieves grounded agricultural research and answers
    """
    # 1. Vision Prediction
    vision_model, err = load_vision_model()
    assert err is None
    assert vision_model is not None

    leaf_img = Image.new("RGB", (224, 224), color=(60, 110, 40))
    prediction_result = predict_leaf_disease(vision_model, leaf_img, lang="en")

    assert prediction_result["success"] is True
    pred_class = prediction_result["predicted_class"]
    confidence = prediction_result["confidence"]

    # 2. Store leaf context (simulating Streamlit session state)
    leaf_context = {
        "success": True,
        "predicted_class": pred_class,
        "confidence": confidence,
        "display_name": prediction_result["display_name"]
    }

    # 3. Pass to Chatbot
    rag_bot = CoffeeRAGChatbot()
    assert rag_bot.is_ready()

    user_query = f"My coffee leaf image was classified as {pred_class}. What are the symptoms and prevention methods?"
    response = rag_bot.answer_question(
        question=user_query,
        leaf_context=leaf_context,
        lang="en"
    )

    # 4. Verify output
    assert "answer" in response
    assert len(response["sources"]) > 0
    assert pred_class.lower() in response["answer"].lower() or "coffee" in response["answer"].lower()
    assert response["leaf_context"] == leaf_context


def test_amharic_end_to_end_workflow():
    """
    Test end-to-end workflow in Amharic:
    1. Vision prediction with Amharic display names
    2. Amharic user question
    3. RAG retrieval using translated query
    4. Amharic response generation with sources
    """
    vision_model, _ = load_vision_model()
    leaf_img = Image.new("RGB", (224, 224), color=(40, 130, 40))
    prediction = predict_leaf_disease(vision_model, leaf_img, lang="am")

    assert prediction["success"] is True
    assert translation_service.is_amharic(prediction["display_name"])

    rag_bot = CoffeeRAGChatbot()
    am_query = "የቡና ቅጠል ዝገት ምልክቶች ምንድን ናቸው?"
    am_res = rag_bot.answer_question(question=am_query, lang="am")

    assert len(am_res["sources"]) > 0
    assert translation_service.is_amharic(am_res["answer"])
