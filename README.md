# ☕ Ethiopian Coffee AI Assistant

**Integrated Computer Vision, Retrieval-Augmented Generation (RAG) Chatbot, and Amharic Language Support for Ethiopian Coffee Farmers and Agronomists**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5%2B-orange.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Overview

The **Ethiopian Coffee AI Assistant** is a unified, deployment-ready agricultural intelligence platform. It merges two independently developed artificial intelligence systems into one cohesive decision-support tool:
1. **Computer Vision Leaf Disease Classifier:** A PyTorch ResNet-18 deep convolutional neural network that analyzes coffee leaf images to detect early pathogens and insect damage.
2. **Retrieval-Augmented Generation (RAG) Chatbot:** An evidence-grounded conversational agent querying a ChromaDB vector database of verified Ethiopian coffee agricultural literature and extension manuals.
3. **Amharic Language Localization (አማርኛ):** Bi-directional translation and Ethiopic Unicode interface enabling smallholder farmers across Ethiopia to interact in their native language.

Critically, the two AI modules are deeply integrated: the computer vision diagnosis automatically transfers into the chatbot's conversational context, allowing users to click **"Ask AI About This Disease"** and immediately receive evidence-based agronomic guidance for the specific scanned leaf.

---

## 2. Problem Statement

Ethiopia is the biological birthplace of *Coffea arabica*, where coffee generates over 30% of the country's foreign export revenue and supports the livelihoods of more than 15 million smallholder farmers. However, smallholder producers face devastating crop losses (up to 40–100% in severe outbreaks) caused by foliar diseases such as **Coffee Leaf Rust** (*Hemileia vastatrix*), **Phoma Leaf Spot** (*Phoma tarda*), and pest infestations like the **Coffee Leaf Miner** (*Leucoptera coffeella*).

Smallholder farmers often lack immediate access to certified agronomists or plant pathology laboratories, and available digital tools typically suffer from critical shortcomings:
- Existing tools present disconnected image classification results without contextual agricultural guidance.
- Chatbots frequently hallucinate chemical dosages or recommendations without scientific evidence.
- Language barriers exclude Ethiopian rural communities when platforms operate only in English.

---

## 3. System Objectives

- **Unified Diagnostic Pipeline:** Deliver an intuitive, single-page and multi-tab interface where visual diagnostics seamlessly inform conversational advice.
- **Evidence-Grounded Retrieval:** Ground all conversational responses in 1,400+ peer-reviewed chunks of Ethiopian agricultural research using dense vector embeddings, eliminating hallucinations.
- **Native Amharic Support:** Support native Ethiopic typography (`U+1200 - U+137F`), agricultural lexicon mapping, and dual-language UI toggling without state loss.
- **Resource Efficiency & Reliability:** Operate efficiently on CPU-only standard personal computers and low-memory cloud tiers (such as Streamlit Community Cloud) with lazy loading and cached singletons.

---

## 4. System Architecture

```
                                  [ User / Farmer ]
                                    │             │
                             (Leaf Image)   (Query: EN / AM)
                                    │             │
                                    ▼             ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT WEB INTERFACE                         │
│   • Responsive Coffee-Themed UI           • Session State Controller   │
│   • English / አማርኛ Language Switcher      • Diagnostic Cards & Charts  │
└───────────────────┬─────────────────────────────────────┬──────────────┘
                    │                                     │
                    ▼                                     ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│       COMPUTER VISION MODULE         │  │     AMHARIC TRANSLATION      │
│  • PIL Image Validation & Preprocess │  │  • Ethiopic Script Detector  │
│  • ResNet-18 Deep Neural Network     │  │  • Agricultural Query Mapper │
│  • Softmax Scores & Probabilities    │  │  • Offline Lexicon Fallback  │
└───────────────────┬──────────────────┘  └──────────────┬───────────────┘
                    │                                    │
                    │   [ Disease Context Transfer ]     │
                    └─────────────────┬──────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          RAG CHATBOT MODULE                            │
│  • Query Vectorization via BAAI/bge-small-en-v1.5                      │
│  • Dense Semantic Retrieval from ChromaDB (1,418 research chunks)      │
│  • Grounded Synthesis Engine (Symptoms, Prevention, IPM, Advisory)     │
│  • Verifiable Source Citation & Metadata Display                       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Computer Vision Module (ResNet-18)

### Architecture
The visual classifier reconstructs the exact architecture from the original PyTorch training checkpoint:
- **Feature Extractor:** `torchvision.models.resnet18` with ImageNet convolutional backbones.
- **Classification Head (`fc`):**
  ```python
  nn.Sequential(
      nn.Dropout(p=0.3),
      nn.Linear(in_features=512, out_features=512),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.2),
      nn.Linear(in_features=512, out_features=4)
  )
  ```
- **Checkpoint Location:** `models/coffee_leaf_disease_model.pth` (~45.8 MB).

### Target Disease Classes (Class Order):
1. **Index 0 - Leaf Miner (`Leucoptera coffeella`):** Insect larvae excavating serpentine mines inside the leaf blade.
2. **Index 1 - Healthy:** Normal, unblemished, photosynthetic leaf tissue with uniform coloration.
3. **Index 2 - Phoma (`Phoma tarda`):** Dark circular necrotic spots with yellow chlorotic halos, frequent at high altitudes.
4. **Index 3 - Rust (`Hemileia vastatrix`):** Yellow-orange powdery spore pustules on the lower leaf surface.

### Preprocessing & Execution:
- Images are validated, converted to RGB, resized to `(224, 224)`, and normalized using standard ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- Runs within `torch.inference_mode()` for zero-gradient overhead and fast CPU execution.

---

## 6. RAG Chatbot & Knowledge Base

### Retrieval Engine:
- **Vector Store:** ChromaDB persistent storage at `vector_db/chroma.sqlite3`.
- **Collection Name:** `coffee_disease_v2` (containing **1,418 indexed research chunks** extracted from Ethiopian agronomy trials, Jimma Agricultural Research Center publications, and international journal studies).
- **Embedding Model:** `BAAI/bge-small-en-v1.5` (384-dimensional dense semantic vectors, cached locally).

### Evidence-Grounded Answer Synthesis:
The chatbot never hallucinates unsupported chemicals or claims guarantees. Every generated answer is structured into:
- **Disease / Topic:** Identified entity.
- **Explanation:** Agronomic background.
- **Symptoms:** Documented field observations.
- **Causes & Favorable Factors:** Environmental and biological vectors.
- **Prevention & Cultural Management:** Shade regulation, pruning, and organic compost practices.
- **Recommended Management Options:** Non-chemical and approved contact treatments.
- **Agronomic Advisory:** Explicit guidance to consult local Development Agents (DAs).
- **Traceable Source Citations:** Direct display of source documents and chunk IDs.

---

## 7. Amharic Translation & Localization

The translation subsystem (`chatbot/translation.py`) features:
1. **Ethiopic Script Detection:** Regex detection (`[\u1200-\u137F]`) to dynamically classify input language.
2. **Cross-Lingual Query Normalization:** Translates Amharic questions into enriched English agricultural search terms (e.g., mapping `"የቡና ዝገት ምልክቶች"` to `"coffee leaf rust Hemileia vastatrix symptoms"`), ensuring high semantic retrieval accuracy against English research documents.
3. **Bi-Directional Translation Pipeline:** Utilizes an online translation adapter with an offline rule-based agricultural lexicon fallback.
4. **LRU In-Memory Caching:** Instant retrieval for repeated translations without latency or external network dependencies.

---

## 8. Repository Structure

```
Ethiopian-Coffee-AI-Assistant/
├── app.py                      # Main Streamlit web application
├── README.md                   # Comprehensive documentation and project guide
├── requirements.txt            # Consolidated production dependencies
├── .gitignore                  # Git exclusions for venvs, caches, and secrets
├── .env.example                # Example environment variable template
├── run_app.bat                 # Windows one-click application launcher
│
├── detection/                  # Computer Vision subsystem
│   ├── __init__.py
│   ├── model.py                # ResNet-18 model architecture
│   ├── inference.py            # Preprocessing, forward pass & prediction logic
│   └── labels.json             # English and Amharic disease metadata
│
├── chatbot/                    # RAG and translation subsystem
│   ├── __init__.py
│   ├── rag_chatbot.py          # ChromaDB retrieval and grounded synthesis
│   └── translation.py          # Bi-directional English <-> Amharic service
│
├── assets/                     # Visual design assets
│   ├── logo.svg                # Vector coffee branch & cherry logo
│   ├── logo.png                # High-resolution raster logo
│   └── styles.css              # Custom warm coffee palette styling
│
├── models/                     # Trained model storage
│   └── coffee_leaf_disease_model.pth # ResNet-18 trained weights (~45.8 MB)
│
├── vector_db/                  # Persistent ChromaDB storage
│   ├── chroma.sqlite3          # SQLite vector index (1,418 chunks)
│   └── ...                     # Vector index segments
│
└── tests/                      # Automated test suite
    ├── test_vision.py          # Vision unit tests and real model inference
    ├── test_rag.py             # ChromaDB and retrieval unit tests
    ├── test_translation.py     # Amharic script and translation tests
    └── test_integration.py     # End-to-end vision-to-chatbot tests
```

---

## 9. Installation and Setup

### Prerequisites
- Windows, macOS, or Linux
- Python 3.10 or 3.11 installed
- At least 4GB of RAM (8GB recommended)

### Step 1: Clone or Open Repository
```bash
cd "C:\Users\DELL\Documents\All projects\Ethiopian Coffee AI\Ethiopian-Coffee-AI-Assistant"
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Consolidated Dependencies
```bash
pip install -r requirements.txt
```

---

## 10. Running the Application

### Option A: One-Click Windows Launcher
Double-click `run_app.bat` or execute in terminal:
```cmd
run_app.bat
```

### Option B: Direct Streamlit Command
```bash
streamlit run app.py
```
The application will launch in your default browser at `http://localhost:8501`.

---

## 11. Automated Testing

The project includes an automated test suite verifying model loading, tensor shapes, vector retrieval, Amharic localization, and end-to-end vision-to-chatbot flow:

```bash
python -m pytest tests -v
```

### Test Results Summary:
- `test_vision.py`: **6 passed** (checkpoint resolution, ResNet-18 architecture, RGB preprocessing, invalid image safety, real leaf inference, Amharic metadata output).
- `test_rag.py`: **5 passed** (ChromaDB connection, chunk verification, real coffee rust retrieval, grounded answer formatting, leaf context integration).
- `test_translation.py`: **5 passed** (Ethiopic Unicode detection, UI text localization, Amharic query expansion, offline lexicon fallback, translation caching).
- `test_integration.py`: **2 passed** (End-to-end Vision -> Context Transfer -> Chatbot answer, Amharic end-to-end workflow).
- **Total: 18 passed in 86 seconds (100% pass rate).**

---

## 12. Deployment Guide

### Deployment on Streamlit Community Cloud:
1. Initialize a Git repository in `Ethiopian-Coffee-AI-Assistant`:
   ```bash
   git init
   git add .
   git commit -m "Deploy Ethiopian Coffee AI Assistant"
   ```
2. Push the repository to GitHub.
3. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
4. Select your repository, set the main file path to `app.py`, and click **Deploy**.
5. The lightweight memory footprint (~1.2GB peak during inference) is fully compatible with Streamlit Community Cloud's standard tier.

---

## 13. Known Limitations & Responsible-Use Statement

- **Model Scope:** The visual model was trained on 4 specific classes (`Leaf Miner`, `Healthy`, `Phoma`, `Rust`). Images of non-coffee leaves, severe nutrient deficiencies, or other pathogens (e.g., Anthracnose) will be mapped to the closest feature representation and should be interpreted with caution.
- **Advisory Nature:** All AI outputs constitute decision-support guidance and educational material, not certified plant pathology diagnoses.
- **Chemical Safety:** Never apply restricted synthetic pesticides without on-site consultation from a certified agricultural extension officer or local Development Agent (DA).

---

## 14. Acknowledgments

- **Jimma Agricultural Research Center (JARC)** and the **Ethiopian Institute of Agricultural Research (EIAR)** for agronomic literature and research publications.
- **PyTorch & Torchvision Teams** for foundational deep learning frameworks.
- **ChromaDB & Sentence-Transformers** for open-source vector retrieval infrastructure.
