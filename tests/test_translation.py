"""
Automated unit tests for Amharic translation module.
Tests Ethiopic script detection, agricultural query expansion,
offline dictionary fallback, UI text localization, and error handling.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot.translation import TranslationService, translation_service, UI_STRINGS


def test_amharic_script_detection():
    """Verify accurate detection of Ethiopic Unicode characters."""
    assert TranslationService.is_amharic("የቡና ቅጠል ዝገት") is True
    assert TranslationService.is_amharic("ሰላም ለሁላችሁ") is True
    assert TranslationService.is_amharic("Coffee Leaf Rust") is False
    assert TranslationService.is_amharic("12345!@#$") is False
    assert TranslationService.is_amharic("") is False


def test_ui_localization_keys():
    """Verify that all core UI keys have non-empty English and Amharic translations."""
    required_keys = [
        "app_title", "app_tagline", "nav_home", "nav_detection",
        "nav_assistant", "nav_about", "upload_header", "ask_ai_about_disease"
    ]
    for key in required_keys:
        en_val = translation_service.get_ui_text(key, lang="en")
        am_val = translation_service.get_ui_text(key, lang="am")

        assert en_val and len(en_val) > 0, f"Missing English UI string for '{key}'"
        assert am_val and len(am_val) > 0, f"Missing Amharic UI string for '{key}'"
        assert TranslationService.is_amharic(am_val) is True, f"Amharic string for '{key}' missing Ethiopic chars: {am_val}"


def test_amharic_query_expansion_for_retrieval():
    """Verify that Amharic questions are enriched with English agricultural keywords for ChromaDB."""
    query = "የቡና ዝገት በሽታ ምልክቶች ምንድን ናቸው?"
    expanded = translation_service.translate_amharic_query_for_retrieval(query)

    assert len(expanded) > 0
    # Must map to English agricultural search terms
    assert any(term in expanded.lower() for term in ["rust", "hemileia", "coffee", "symptom"])


def test_offline_lexicon_translation_fallback():
    """Verify fallback translation rules work even when offline."""
    svc = TranslationService(enable_web_api=False)

    text = "Disease / Topic: Coffee Leaf Rust\nSymptoms:\nTreatment:"
    translated = svc.translate_text(text, source_lang="en", target_lang="am")

    assert "በሽታ / ርዕስ:" in translated
    assert "ምልክቶች:" in translated
    assert "የቡና ቅጠል ዝገት" in translated


def test_translation_caching():
    """Verify identical queries are cached and served from memory."""
    svc = TranslationService(enable_web_api=False)
    input_str = "Coffee Leaf Miner"
    res1 = svc.translate_text(input_str, target_lang="am")
    res2 = svc.translate_text(input_str, target_lang="am")

    assert res1 == res2
    assert (input_str, "auto", "am") in svc.cache
