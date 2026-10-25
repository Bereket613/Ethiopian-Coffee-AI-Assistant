"""
Modular RAG Chatbot engine for Ethiopian Coffee AI Assistant.
Integrates ChromaDB persistent storage, BAAI/bge-small-en-v1.5 embeddings,
grounded multi-backend answer generation, and leaf diagnosis context integration.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
import torch

from .translation import translation_service

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "coffee_disease_v2"
FALLBACK_COLLECTION_NAME = "coffee_disease"


def resolve_vector_db_path(custom_path: Optional[str] = None) -> Path:
    """Resolve vector_db directory from multiple candidate locations."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    if os.environ.get("CHROMA_DB_PATH"):
        candidates.append(Path(os.environ["CHROMA_DB_PATH"]))

    base_dir = Path(__file__).resolve().parent.parent
    candidates.extend([
        base_dir / "vector_db",
        base_dir.parent / "Coffee-RAG-Chatbot" / "vector_db",
        Path("vector_db"),
        Path("Coffee-RAG-Chatbot/vector_db")
    ])

    for path in candidates:
        if path.is_dir() and (path / "chroma.sqlite3").is_file():
            return path

    # Fallback to base_dir / "vector_db" even if not yet created
    return base_dir / "vector_db"


class CoffeeRAGChatbot:
    """
    RAG Chatbot module querying local ChromaDB vector database with
    BAAI/bge-small-en-v1.5 and generating grounded agricultural responses.
    """
    def __init__(self, db_path: Optional[str] = None, collection_name: str = DEFAULT_COLLECTION_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.db_path = resolve_vector_db_path(db_path)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embed_model = None

        self._init_database()
        self._init_embedding_model()

    def _init_database(self):
        """Initialize ChromaDB client and collection."""
        try:
            if not self.db_path.exists():
                logger.warning(f"Vector DB path does not exist: {self.db_path}")
                return

            self.client = chromadb.PersistentClient(path=str(self.db_path))
            existing_cols = [c.name for c in self.client.list_collections()]

            if self.collection_name in existing_cols:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded collection '{self.collection_name}' with {self.collection.count()} chunks.")
            elif FALLBACK_COLLECTION_NAME in existing_cols:
                logger.info(f"Collection '{self.collection_name}' not found. Falling back to '{FALLBACK_COLLECTION_NAME}'.")
                self.collection = self.client.get_collection(name=FALLBACK_COLLECTION_NAME)
            else:
                logger.warning(f"Neither '{self.collection_name}' nor '{FALLBACK_COLLECTION_NAME}' exists. Available: {existing_cols}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}", exc_info=True)
            self.collection = None

    def _init_embedding_model(self):
        """Load BAAI/bge-small-en-v1.5 embedding model with caching."""
        try:
            logger.info(f"Loading embedding model BAAI/bge-small-en-v1.5 on {self.device}...")
            self.embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=self.device)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}", exc_info=True)
            self.embed_model = None

    def is_ready(self) -> bool:
        """Check if both ChromaDB collection and embedding model are ready."""
        return self.collection is not None and self.embed_model is not None

    def retrieve_chunks(self, question: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant document chunks from ChromaDB for a given query.
        Returns a list of dicts with document, id, and score.
        """
        if not self.is_ready():
            logger.warning("RAG Chatbot is not ready for retrieval.")
            return []

        try:
            q_emb = self.embed_model.encode([question], convert_to_numpy=True)[0]
            results = self.collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=k
            )

            chunks = []
            if results and results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                ids = results["ids"][0] if results.get("ids") else [f"doc_{i}" for i in range(len(docs))]
                distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)

                for doc, id_, dist, meta in zip(docs, ids, distances, metas):
                    # Clean source name from chunk filename
                    source_title = id_.split("_chunks.txt")[0] if "_chunks.txt" in id_ else id_
                    chunks.append({
                        "id": id_,
                        "source": source_title,
                        "text": doc.strip(),
                        "distance": dist,
                        "metadata": meta
                    })
            return chunks

        except Exception as e:
            logger.error(f"Error during chunk retrieval: {e}", exc_info=True)
            return []

    def answer_question(
        self,
        question: str,
        leaf_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        lang: str = "en",
        k: int = 3
    ) -> Dict[str, Any]:
        """
        Main query processing pipeline:
        1. Query translation & enrichment if input is in Amharic
        2. Contextual enrichment with computer vision leaf diagnosis (if present)
        3. Vector retrieval from ChromaDB
        4. Grounded answer generation
        5. Translation to requested output language (Amharic or English)
        """
        is_user_amharic = translation_service.is_amharic(question) or lang == "am"

        # Step 1: Prepare search query for English vector DB
        search_query = question
        if is_user_amharic:
            search_query = translation_service.translate_amharic_query_for_retrieval(question)

        # Step 2: Inject computer vision prediction if leaf_context is active
        context_prefix = ""
        if leaf_context and leaf_context.get("predicted_class"):
            pred_class = leaf_context["predicted_class"]
            conf = leaf_context.get("confidence", 0.0)
            if pred_class.lower() not in search_query.lower():
                search_query = f"{pred_class} coffee disease {search_query}"
            context_prefix = f"[Computer Vision Context: Scanned Leaf detected as '{pred_class}' ({conf}% model score)]\n"

        # Step 3: Retrieve relevant chunks
        chunks = self.retrieve_chunks(search_query, k=k)

        # Step 4: Generate grounded answer
        if not chunks:
            if is_user_amharic:
                empty_msg = "በዕውቀት ማህደሩ ውስጥ ስለዚህ ጉዳይ በቂ መረጃ ማግኘት አልተቻለም። እባክዎ ስለ ቡና ቅጠል ዝገት፣ ፎማ፣ ቅጠል ሰርጣቂ ወይም የቡና ፍሬ በሽታ ምልክቶችና አያያዝ ይጠይቁ።"
            else:
                empty_msg = "I could not find sufficient grounded evidence in the agricultural research database for this specific inquiry. Please try asking about coffee leaf rust, phoma, leaf miner, or coffee berry disease management."
            return {
                "answer": empty_msg,
                "sources": [],
                "leaf_context": leaf_context,
                "language": "am" if is_user_amharic else "en"
            }

        grounded_answer = self._synthesize_grounded_answer(
            question=question,
            chunks=chunks,
            leaf_context=leaf_context,
            lang=lang
        )

        # Step 5: Translate to Amharic if requested
        if is_user_amharic:
            final_answer = translation_service.translate_text(
                grounded_answer,
                source_lang="en",
                target_lang="am"
            )
        else:
            final_answer = grounded_answer

        return {
            "answer": final_answer,
            "sources": chunks,
            "leaf_context": leaf_context,
            "language": "am" if is_user_amharic else "en"
        }

    def _synthesize_grounded_answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        leaf_context: Optional[Dict[str, Any]] = None,
        lang: str = "en"
    ) -> str:
        """
        Synthesizes a structured, evidence-grounded answer directly from retrieved chunks.
        Never fabricates facts, dosages, or unsupported chemical formulas.
        """
        combined_text = "\n\n".join([c["text"] for c in chunks])

        # Identify disease focus
        diseases = ["Coffee Leaf Rust", "Leaf Miner", "Phoma", "Coffee Berry Disease", "Coffee Wilt Disease"]
        detected_topic = "Coffee Plant Health"
        if leaf_context and leaf_context.get("predicted_class"):
            detected_topic = leaf_context["predicted_class"]
        else:
            for d in diseases:
                if d.lower() in question.lower() or d.lower() in combined_text.lower():
                    detected_topic = d
                    break

        # Extract relevant key points from retrieved research text
        sentences = [s.strip() for s in re.split(r'\. |\n', combined_text) if len(s.strip()) > 25]

        symptom_cues = ["symptom", "lesion", "spot", "yellow", "orange", "powdery", "tunnel", "mine", "defoliation", "damage"]
        cause_cues = ["cause", "fungus", "larva", "larvae", "insect", "hemileia", "pathogen", "infestation", "weather", "rain"]
        prevention_cues = ["prevent", "control", "shade", "prun", "resistant", "variety", "drainage", "mulch", "spacing", "clean"]
        treatment_cues = ["treat", "fungicide", "spray", "copper", "organic", "biological", "parasitoid", "management"]

        found_symptoms = []
        found_causes = []
        found_prevention = []
        found_treatments = []

        for s in sentences:
            s_lower = s.lower()
            if any(c in s_lower for c in symptom_cues) and len(found_symptoms) < 3:
                if s not in found_symptoms:
                    found_symptoms.append(s)
            elif any(c in s_lower for c in cause_cues) and len(found_causes) < 2:
                if s not in found_causes:
                    found_causes.append(s)
            elif any(c in s_lower for c in prevention_cues) and len(found_prevention) < 3:
                if s not in found_prevention:
                    found_prevention.append(s)
            elif any(c in s_lower for c in treatment_cues) and len(found_treatments) < 2:
                if s not in found_treatments:
                    found_treatments.append(s)

        # Build clean, farmer-friendly response structure
        parts = [f"### Disease / Topic: {detected_topic}\n"]

        # If vision context was present, mention it respectfully
        if leaf_context and leaf_context.get("predicted_class"):
            pred = leaf_context["predicted_class"]
            conf = leaf_context.get("confidence", 0.0)
            parts.append(f"**Vision Diagnostic Context:** The scanned leaf was classified as **{pred}** with a model confidence score of **{conf}%**. The research findings below address this condition.\n")

        parts.append(f"**Explanation:**\nBased on the retrieved agricultural research, {detected_topic.lower()} impacts Arabica coffee production in Ethiopia. Proper identification and timely canopy and soil management are critical to protect yield.\n")

        if found_symptoms:
            parts.append("**Documented Symptoms:**")
            for item in found_symptoms[:2]:
                parts.append(f"• {item}")
            parts.append("")

        if found_causes:
            parts.append("**Causes & Favorable Factors:**")
            for item in found_causes[:2]:
                parts.append(f"• {item}")
            parts.append("")

        if found_prevention:
            parts.append("**Prevention & Cultural Practices:**")
            for item in found_prevention[:2]:
                parts.append(f"• {item}")
            parts.append("")

        if found_treatments:
            parts.append("**Recommended Management Options:**")
            for item in found_treatments[:2]:
                parts.append(f"• {item}")
            parts.append("")

        # Responsible agricultural disclaimer
        parts.append("**Agronomic Advisory:** For any chemical or organic spray applications, always consult your local development agent (DA) or agricultural extension officer to verify locally approved products and safety intervals.")

        return "\n".join(parts)
