"""
Translation service for Ethiopian Coffee AI Assistant.
Provides bidirectional English <-> Amharic translation for UI,
search query normalization for vector retrieval, and answer translation.
Includes offline agricultural lexicon fallback and caching.
"""

import re
import urllib.request
import urllib.parse
import json
import logging
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

# Core UI Text Dictionary
UI_STRINGS: Dict[str, Dict[str, str]] = {
    "app_title": {
        "en": "ETHIOPIAN COFFEE AI ASSISTANT",
        "am": "የኢትዮጵያ ቡና አርቴፊሻል ኢንተለጀንስ ረዳት"
    },
    "app_tagline": {
        "en": "Smarter Coffee Farming Through AI",
        "am": "በአርቴፊሻል ኢንተለጀንስ የተሻለ የቡና እርሻ"
    },
    "nav_home": {
        "en": "Home",
        "am": "መነሻ ገጽ"
    },
    "nav_detection": {
        "en": "Disease Detection",
        "am": "የበሽታ መለያ"
    },
    "nav_assistant": {
        "en": "AI Coffee Assistant",
        "am": "የቡና ረዳት ውይይት"
    },
    "nav_about": {
        "en": "About the Project",
        "am": "ስለ ፕሮጀክቱ"
    },
    "language_selector_label": {
        "en": "Language / ቋንቋ",
        "am": "ቋንቋ / Language"
    },
    "welcome_header": {
        "en": "Empowering Ethiopian Coffee Farmers with Intelligent Diagnostics",
        "am": "የኢትዮጵያ ቡና አምራቾችን በዘመናዊ የቴክኖሎጂ እውቀት ማብቃት"
    },
    "welcome_subheader": {
        "en": "Integrated computer vision for instant leaf disease classification combined with a grounded RAG chatbot for evidence-based agricultural advice.",
        "am": "በቅጠል ምስል በሽታን በፍጥነት የሚለይ የኮምፒውተር እይታ እና በማስረጃ የተደገፈ የግብርና መረጃ የሚሰጥ የውይይት ረዳት ቅንጅት።"
    },
    "action_detect": {
        "en": "Scan Coffee Leaf",
        "am": "የቡና ቅጠል ይመርምሩ"
    },
    "action_chat": {
        "en": "Consult AI Assistant",
        "am": "ከረዳቱ ጋር ይወያዩ"
    },
    "responsible_use_banner": {
        "en": "Notice: This AI system is an educational and diagnostic support tool designed to assist farmers and agronomists. For chemical applications or severe outbreaks, always verify with local agricultural extension officers.",
        "am": "ማሳሰቢያ፡ ይህ አርቴፊሻል ኢንተለጀንስ አርሶ አደሮችን እና የግብርና ባለሙያዎችን ለመደገፍ የተዘጋጀ ነው። ኬሚካሎችን ከመጠቀምዎ በፊት ሁልጊዜ የአካባቢዎን የግብርና ኤክስቴንሽን ባለሙያ ያማክሩ።"
    },
    "upload_header": {
        "en": "Upload Coffee Leaf Image",
        "am": "የቡና ቅጠል ምስል ያስገቡ"
    },
    "upload_instructions": {
        "en": "Upload a clear, well-lit photo of an individual coffee leaf (JPG, JPEG, or PNG).",
        "am": "ግልጽ የሆነና በደንብ የሚታይ ነጠላ የቡና ቅጠል ፎቶ ያስገቡ (JPG, JPEG ወይም PNG)።"
    },
    "upload_button_label": {
        "en": "Choose an image file",
        "am": "የምስል ፋይል ይምረጡ"
    },
    "preview_heading": {
        "en": "Leaf Image Preview",
        "am": "የተመረጠው ምስል እይታ"
    },
    "analyzing_spinner": {
        "en": "Analyzing leaf morphology and disease patterns...",
        "am": "የቅጠሉን ሁኔታ እና የበሽታ ምልክቶች በመመርመር ላይ..."
    },
    "prediction_result_title": {
        "en": "Diagnostic Prediction",
        "am": "የምርመራ ውጤት"
    },
    "model_score_label": {
        "en": "Model Confidence Score",
        "am": "የሞዴል እርግጠኝነት መጠን"
    },
    "probability_distribution_title": {
        "en": "Output Score Distribution Across Classes",
        "am": "የውጤት ክፍፍል በየበሽታው ዓይነት"
    },
    "ask_ai_about_disease": {
        "en": "Ask AI About This Disease",
        "am": "ስለዚህ በሽታ ረዳቱን ይጠይቁ"
    },
    "upload_another_button": {
        "en": "Upload Another Leaf",
        "am": "ሌላ ምስል ይመረምሩ"
    },
    "chat_input_placeholder": {
        "en": "Ask any question about coffee diseases, symptoms, or organic treatments...",
        "am": "ስለ ቡና በሽታዎች፣ ምልክቶች ወይም ተፈጥሯዊ ህክምናዎች ጥያቄዎን ይጠይቁ..."
    },
    "sources_expander_title": {
        "en": "Retrieved Grounding Sources (ChromaDB)",
        "am": "የተገኙ የማስረጃ ሰነዶች (ከዕውቀት ማህደሩ)"
    },
    "chat_context_badge": {
        "en": "Active Leaf Context",
        "am": "ወቅታዊ የቅጠል ምርመራ መረጃ"
    },
    "clear_chat_button": {
        "en": "Clear Conversation",
        "am": "ውይይቱን አጽዳ"
    },
    "example_questions_header": {
        "en": "Suggested Questions",
        "am": "የተመረጡ የናሙና ጥያቄዎች"
    },
    "q1": {
        "en": "What are the early symptoms of Coffee Leaf Rust and how does it spread?",
        "am": "የቡና ቅጠል ዝገት የመጀመሪያ ምልክቶች ምንድን ናቸው? እንዴትስ ይሰራጫል?"
    },
    "q2": {
        "en": "How can smallholder farmers manage Coffee Berry Disease organically?",
        "am": "አነስተኛ አርሶ አደሮች የቡና ፍሬ በሽታን በተፈጥሯዊ መንገድ እንዴት መከላከል ይችላሉ?"
    },
    "q3": {
        "en": "What shade tree species are recommended for Arabica coffee in Ethiopia?",
        "am": "በኢትዮጵያ ውስጥ ለአረቢካ ቡና ተስማሚ የሆኑ የጥላ ዛፎች የትኞቹ ናቸው?"
    },
    "q4": {
        "en": "How do you distinguish Phoma leaf spot from leaf scorch or sun damage?",
        "am": "የፎማ ቅጠል ነጠብጣብ በሽታን ከፀሐይ ቃጠሎ እንዴት መለየት ይቻላል?"
    },
    "disclaimer_label": {
        "en": "Agricultural Advisory Disclaimer",
        "am": "የግብርና ምክር ማስተባበያ"
    },
    "disclaimer_body": {
        "en": "Predictions and generated responses are derived from computer vision inference and retrieved research literature. They do not constitute an agronomic guarantee. Always adhere to local pesticide safety regulations and certified extension protocols.",
        "am": "የቀረቡት ውጤቶች እና መረጃዎች ከኮምፒውተር እይታ እና ከተመረጡ የምርምር ሰነዶች የተገኙ ናቸው። የኬሚካል አጠቃቀምን በተመለከተ ሁልጊዜ የአካባቢዎን የግብርና መመሪያዎች ይከተሉ።"
    }
}

# Agricultural Domain Translation Lexicon for English <-> Amharic
AGRICULTURAL_LEXICON = [
    # Diseases
    (r"\bcoffee leaf rust\b", "የቡና ቅጠል ዝገት", "Coffee Leaf Rust"),
    (r"\bleaf rust\b", "የቅጠል ዝገት", "Leaf Rust"),
    (r"\bleaf miner\b", "የቅጠል ሰርጣቂ ተባይ", "Leaf Miner"),
    (r"\bphoma\b", "ፎማ", "Phoma"),
    (r"\bphoma leaf spot\b", "የፎማ ቅጠል ነጠብጣብ በሽታ", "Phoma Leaf Spot"),
    (r"\bcoffee berry disease\b", "የቡና ፍሬ በሽታ (ሲ.ቢ.ዲ)", "Coffee Berry Disease"),
    (r"\bcoffee wilt disease\b", "የቡና ዛፍ አውራ አደር / መድረቅ በሽታ", "Coffee Wilt Disease"),
    (r"\bhealthy\b", "ጤናማ", "Healthy"),

    # Agronomy terms
    (r"\bsymptoms?\b", "ምልክቶች", "symptoms"),
    (r"\bprevention\b", "መከላከያ", "prevention"),
    (r"\btreatment\b", "ህክምና", "treatment"),
    (r"\bmanagement\b", "አያያዝ እና መቆጣጠሪያ ዘዴዎች", "management"),
    (r"\bcauses?\b", "ምክንያቶች", "causes"),
    (r"\bfungus\b|\bfungal\b", "ፈንገስ", "fungus"),
    (r"\blesions?\b", "ቁስሎች", "lesions"),
    (r"\bpustules?\b", "ነጠብጣቦች", "pustules"),
    (r"\bspores?\b", "ስፖሮች (የፈንገስ ዘሮች)", "spores"),
    (r"\bshade trees?\b", "የጥላ ዛፎች", "shade trees"),
    (r"\bpruning\b", "ቅርንጫፍ መግረዝ", "pruning"),
    (r"\bcopper spray\b|\bcopper fungicide\b", "የኮፐር ጸረ-ፈንገስ", "copper fungicide"),
    (r"\bfertilizer\b|\bcompost\b", "ኮምፖስት እና ማዳበሪያ", "organic fertilizer"),
    (r"\bharvest\b", "ምርት ስብሰባ", "harvest"),
    (r"\byield\b", "ምርት መጠን", "yield"),
]

# Amharic query mapping to English keywords for ChromaDB retrieval
AMHARIC_TO_ENGLISH_SEARCH_TERMS = {
    "ዝገት": "coffee leaf rust Hemileia vastatrix",
    "የቡና ዝገት": "coffee leaf rust Hemileia vastatrix symptoms control",
    "ሰርጣቂ": "leaf miner Leucoptera coffeella insect",
    "ቅጠል ሰርጣቂ": "coffee leaf miner Leucoptera control damage",
    "ፎማ": "Phoma tarda leaf spot fungal lesions",
    "የፎማ በሽታ": "Phoma leaf spot management high altitude",
    "ፍሬ በሽታ": "coffee berry disease Colletotrichum kahawae",
    "ሲቢዲ": "coffee berry disease Colletotrichum kahawae",
    "መድረቅ": "coffee wilt disease Gibberella xylarioides",
    "ምልክት": "symptoms identification",
    "ምልክቶች": "symptoms visual signs",
    "መከላከያ": "prevention control cultural practices",
    "ህክምና": "treatment fungicide management spray",
    "ጥላ ዛፍ": "shade tree canopy Cordia africana Millettia",
    "መግረዝ": "pruning canopy management aeration",
    "ኮምፖስት": "compost organic soil nutrition",
    "ፈንገስ": "fungus fungal pathogen spore"
}


class TranslationService:
    """
    Bidirectional translation service with web API fallback,
    in-memory caching, and domain-specific agricultural dictionary.
    """
    def __init__(self, enable_web_api: bool = True):
        self.enable_web_api = enable_web_api
        self.cache: Dict[Tuple[str, str, str], str] = {}

    @staticmethod
    def is_amharic(text: str) -> bool:
        """Detect if text contains Ethiopic Unicode characters (U+1200 - U+137F)."""
        if not text:
            return False
        ethiopic_pattern = re.compile(r'[\u1200-\u137F]')
        return bool(ethiopic_pattern.search(text))

    def get_ui_text(self, key: str, lang: str = "en") -> str:
        """Retrieve localized UI text with safe fallback."""
        entry = UI_STRINGS.get(key, {})
        lang_key = "am" if lang == "am" else "en"
        return entry.get(lang_key, entry.get("en", key))

    def translate_amharic_query_for_retrieval(self, amharic_query: str) -> str:
        """
        Transforms an Amharic query into an enriched English search query
        optimized for semantic similarity retrieval in ChromaDB.
        """
        if not self.is_amharic(amharic_query):
            return amharic_query

        # Extract mapped agricultural keywords
        matched_terms = []
        for am_word, en_term in AMHARIC_TO_ENGLISH_SEARCH_TERMS.items():
            if am_word in amharic_query:
                matched_terms.append(en_term)

        # Also attempt online translation if enabled
        online_trans = self._translate_online(amharic_query, source="am", target="en")
        if online_trans and len(online_trans.strip()) > 3:
            if matched_terms:
                return f"{online_trans} {' '.join(matched_terms)}"
            return online_trans

        # If offline and matched terms exist, build query from matched terms
        if matched_terms:
            return f"Coffee disease {' '.join(matched_terms)}"

        return amharic_query

    def translate_text(self, text: str, source_lang: str = "auto", target_lang: str = "am") -> str:
        """
        Translates text between English and Amharic with caching.
        """
        if not text or not text.strip():
            return text

        cache_key = (text.strip(), source_lang, target_lang)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Check if already in target script
        if target_lang == "am" and self.is_amharic(text) and not re.search(r'[a-zA-Z]{5,}', text):
            return text
        if target_lang == "en" and not self.is_amharic(text):
            return text

        # Attempt online translation
        translated = self._translate_online(text, source=source_lang, target=target_lang)
        if translated:
            self.cache[cache_key] = translated
            return translated

        # Offline fallback
        if target_lang == "am":
            fallback = self._apply_lexicon_en_to_am(text)
        else:
            fallback = self._apply_lexicon_am_to_en(text)

        self.cache[cache_key] = fallback
        return fallback

    def _translate_online(self, text: str, source: str = "auto", target: str = "am") -> Optional[str]:
        """Query free Google Translate web endpoint with timeout and exception safety."""
        if not self.enable_web_api:
            return None

        # Clean source and target codes
        sl = "auto" if source == "auto" else ("am" if source == "am" else "en")
        tl = "am" if target == "am" else "en"

        try:
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={sl}&tl={tl}&dt=t&q={urllib.parse.quote(text)}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=4) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw)
                if data and isinstance(data, list) and data[0]:
                    translated_segments = [seg[0] for seg in data[0] if seg and seg[0]]
                    result = "".join(translated_segments)
                    return result
        except Exception as e:
            logger.debug(f"Online translation unavailable: {e}")
            return None

    def _apply_lexicon_en_to_am(self, text: str) -> str:
        """Apply rule-based agricultural replacements for English to Amharic."""
        res = text
        # Translate common structural headers
        headers_map = [
            ("Disease / Topic:", "በሽታ / ርዕስ:"),
            ("Explanation:", "ማብራሪያ:"),
            ("Symptoms:", "ምልክቶች:"),
            ("Causes:", "ምክንያቶች:"),
            ("Prevention:", "የመከላከያ ዘዴዎች:"),
            ("Treatment:", "የህክምና እና አያያዝ ዘዴዎች:"),
            ("Sources:", "የመረጃ ምንጮች:"),
            ("Recommended Practices:", "የሚመከሩ ተግባራት:"),
            ("Important Note:", "አስፈላጊ ማስታወሻ:"),
        ]
        for en_h, am_h in headers_map:
            res = res.replace(en_h, am_h)

        # Replace domain entities
        for pattern, am_term, _ in AGRICULTURAL_LEXICON:
            res = re.sub(pattern, am_term, res, flags=re.IGNORECASE)

        return res

    def _apply_lexicon_am_to_en(self, text: str) -> str:
        """Apply rule-based agricultural replacements for Amharic to English."""
        res = text
        for _, am_term, en_term in AGRICULTURAL_LEXICON:
            res = res.replace(am_term, en_term)
        return res


# Global singleton instance
translation_service = TranslationService()
