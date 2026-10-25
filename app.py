"""
ETHIOPIAN COFFEE AI ASSISTANT
Unified Web Application combining Computer Vision, RAG Chatbot, and Amharic Support.
"""

import sys
from pathlib import Path

# Ensure application directory is in Python path for portable imports
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

from detection.inference import load_vision_model, predict_leaf_disease, CLASS_NAMES
from chatbot.rag_chatbot import CoffeeRAGChatbot
from chatbot.translation import translation_service

# --- Page Configuration ---
st.set_page_config(
    page_title="Ethiopian Coffee AI Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Custom CSS Styling ---
def load_css():
    css_path = APP_DIR / "assets" / "styles.css"
    if css_path.is_file():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --- Initialize Session State ---
if "lang" not in st.session_state:
    st.session_state.lang = "en"  # "en" or "am"

if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "pending_chat_prompt" not in st.session_state:
    st.session_state.pending_chat_prompt = None

if "nav_target" not in st.session_state:
    st.session_state.nav_target = None


# --- Cached Model & Database Loaders ---
@st.cache_resource(show_spinner=False)
def get_cached_vision_model():
    """Cache-loaded ResNet18 vision model."""
    model, error = load_vision_model()
    return model, error

@st.cache_resource(show_spinner=False)
def get_cached_rag_chatbot():
    """Cache-loaded RAG Chatbot with ChromaDB and SentenceTransformer."""
    try:
        bot = CoffeeRAGChatbot()
        return bot, None
    except Exception as e:
        return None, str(e)


# Helper function for translation
def t(key: str) -> str:
    return translation_service.get_ui_text(key, lang=st.session_state.lang)


# --- Top Brand Header Component ---
def render_brand_header():
    col_logo, col_text, col_lang = st.columns([1, 6, 2])
    
    with col_logo:
        logo_path = APP_DIR / "assets" / "logo.png"
        if logo_path.is_file():
            st.image(str(logo_path), width=85)
        else:
            st.markdown("## 🌿")

    with col_text:
        st.markdown(
            f"""
            <div style="line-height: 1.2; padding-top: 0.2rem;">
                <h1 style="margin: 0; color: #1B4D3E; font-size: 1.95rem; font-weight: 700; letter-spacing: -0.02em;">
                    {t('app_title')}
                </h1>
                <p style="margin: 0.3rem 0 0 0; color: #525B56; font-size: 1.05rem; font-weight: 500;">
                    {t('app_tagline')}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_lang:
        lang_choice = st.selectbox(
            label=t("language_selector_label"),
            options=["English", "አማርኛ (Amharic)"],
            index=0 if st.session_state.lang == "en" else 1,
            key="language_selector_widget"
        )
        new_lang = "am" if "አማርኛ" in lang_choice else "en"
        if new_lang != st.session_state.lang:
            st.session_state.lang = new_lang
            st.rerun()

    st.markdown("<hr style='margin: 0.8rem 0 1.2rem 0; border: none; border-top: 1px solid #E8E2D9;'>", unsafe_allow_html=True)


# --- PAGE 1: HOME ---
def render_home_page():
    st.markdown(
        f"""
        <div class="brand-header-container">
            <h2 class="brand-title">{t('welcome_header')}</h2>
            <p class="brand-subtitle">{t('welcome_subheader')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="feature-card">
                <h3>📷 {t('nav_detection')}</h3>
                <p>{'Analyze single coffee leaf photos using our trained ResNet-18 computer vision model to detect Rust, Phoma, Leaf Miner, or verify leaf health.' if st.session_state.lang == 'en' else 'የሰለጠነ የኮምፒውተር እይታ ሞዴል በመጠቀም የቡና ቅጠል ዝገትን፣ ፎማን፣ የቅጠል ሰርጣቂን ወይም ጤናማ ቅጠልን በፎቶ በፍጥነት ይመርምሩ።'}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button(f"🔍 {t('action_detect')}", key="home_btn_detect", type="primary", use_container_width=True):
            st.session_state.nav_target = t("nav_detection")
            st.rerun()

    with col2:
        st.markdown(
            f"""
            <div class="feature-card">
                <h3>💬 {t('nav_assistant')}</h3>
                <p>{'Interact with our RAG chatbot grounded in verified Ethiopian coffee research papers and Jimma Agricultural Research Center agronomy guides.' if st.session_state.lang == 'en' else 'በተረጋገጡ የኢትዮጵያ የቡና ምርምር ሰነዶች እና በግብርና መመሪያዎች ላይ የተመሰረተ አስተማማኝ የምክር ውይይት ረዳት።'}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button(f"💡 {t('action_chat')}", key="home_btn_chat", type="secondary", use_container_width=True):
            st.session_state.nav_target = t("nav_assistant")
            st.rerun()

    # Advisory Box
    st.markdown(
        f"""
        <div class="advisory-box">
            <div class="advisory-title">⚠️ {t('disclaimer_label')}</div>
            {t('responsible_use_banner')}
        </div>
        """,
        unsafe_allow_html=True
    )


# --- PAGE 2: DISEASE DETECTION ---
def render_detection_page(vision_model, model_error):
    st.markdown(f"## 🌿 {t('nav_detection')}")
    st.markdown(t("upload_instructions"))

    if model_error:
        st.error(f"⚠️ Computer Vision Model Status: {model_error}")
        st.info("The application can continue using the AI Assistant module, but image diagnosis is temporarily disabled.")
        return

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown(f"### 📤 {t('upload_header')}")
        uploaded_file = st.file_uploader(
            label=t("upload_button_label"),
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG. Max file size: 200MB."
        )

        if uploaded_file is not None:
            try:
                preview_image = Image.open(uploaded_file)
                st.image(preview_image, caption=t("preview_heading"), use_container_width=True)

                # Process Image
                with st.spinner(t("analyzing_spinner")):
                    result = predict_leaf_disease(
                        model=vision_model,
                        image_input=uploaded_file,
                        lang=st.session_state.lang
                    )

                if result["success"]:
                    st.session_state.current_prediction = result
                else:
                    st.error(f"Image analysis failed: {result['error']}")

            except Exception as e:
                st.error(f"Could not open image: {e}")

    with col_right:
        pred = st.session_state.current_prediction

        if pred and pred.get("success"):
            predicted_class = pred["predicted_class"]
            confidence = pred["confidence"]
            display_name = pred["display_name"]
            is_healthy = predicted_class == "Healthy"

            card_class = "result-card-healthy" if is_healthy else "result-card-disease"
            badge_class = "badge-healthy" if is_healthy else "badge-disease"
            badge_text = "NORMAL / ጤናማ" if is_healthy else "ATTENTION / ጥንቃቄ"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <span class="result-badge {badge_class}">{badge_text}</span>
                    <div class="prediction-title">{display_name}</div>
                    <div style="margin-top: 0.5rem;">
                        <span class="confidence-chip">🎯 {t('model_score_label')}: {confidence:.1f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Class probability distribution bar chart
            st.markdown(f"#### 📊 {t('probability_distribution_title')}")
            prob_dict = pred["probabilities"]
            
            df_probs = [
                {"Class": c, "Score": prob_dict.get(c, 0.0)}
                for c in CLASS_NAMES
            ]

            fig = px.bar(
                df_probs,
                x="Class",
                y="Score",
                text="Score",
                color="Score",
                color_continuous_scale=["#D8F3DC", "#52B788", "#1B4D3E", "#2A1810"],
                range_y=[0, 105]
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                height=240,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="",
                yaxis_title="Score (%)",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Disease symptoms and management details
            meta = pred.get("metadata", {})
            if meta:
                with st.expander(f"📖 {display_name} - {'Agronomic Guidelines' if st.session_state.lang == 'en' else 'የግብርና መመሪያዎች'}", expanded=True):
                    if meta.get("description"):
                        st.markdown(f"**{'Description' if st.session_state.lang == 'en' else 'ማብራሪያ'}:** {meta['description']}")
                    
                    symptoms = meta.get("symptoms", [])
                    if symptoms:
                        st.markdown(f"**{'Key Symptoms' if st.session_state.lang == 'en' else 'ዋና ዋና ምልክቶች'}:**")
                        for s in symptoms:
                            st.markdown(f"• {s}")

                    management = meta.get("management", [])
                    if management:
                        st.markdown(f"**{'Management & Prevention' if st.session_state.lang == 'en' else 'የመከላከያ እና አያያዝ ዘዴዎች'}:**")
                        for m in management:
                            st.markdown(f"• {m}")

            # INTEGRATION ACTION BUTTON: Ask AI About This Disease
            st.markdown("---")
            if st.button(f"💬 {t('ask_ai_about_disease')}", type="primary", use_container_width=True):
                # Populate the question incorporating the actual vision prediction
                if st.session_state.lang == "am":
                    prompt_text = f"የቡና ቅጠል ምስል በሞዴሉ የተመረመረ ሲሆን ውጤቱም '{display_name}' ({confidence:.1f}%) ሆኗል። በዕውቀት ማህደሩ ላይ በመመስረት የዚህን በሽታ ምልክቶች፣ መከላከያ መንገዶች እና የሚመከሩ የህክምና ዘዴዎችን በዝርዝር አብራራ።"
                else:
                    prompt_text = f"My coffee leaf image was classified as {predicted_class} with a {confidence:.1f}% model score. Based on the agricultural knowledge base, explain the symptoms, prevention methods, and documented management options for this disease."
                
                st.session_state.pending_chat_prompt = prompt_text
                st.session_state.nav_target = t("nav_assistant")
                st.rerun()

        else:
            st.info(
                "Upload a leaf image on the left to view the ResNet-18 classification result, confidence scores, and disease management profile."
                if st.session_state.lang == "en"
                else "የቅጠል ምስል በግራ በኩል ያስገቡ፤ የምርመራ ውጤት፣ እርግጠኝነት እና የበሽታው መከላከያ ዘዴዎች እዚህ ይታያሉ።"
            )


# --- PAGE 3: AI COFFEE ASSISTANT (RAG) ---
def render_assistant_page(rag_bot, rag_error):
    st.markdown(f"## 💬 {t('nav_assistant')}")

    if rag_error:
        st.error(f"⚠️ Knowledge Base Retrieval Error: {rag_error}")
        return

    # Check for active leaf diagnosis context
    pred = st.session_state.current_prediction
    if pred and pred.get("success"):
        p_class = pred.get("predicted_class", "")
        p_conf = pred.get("confidence", 0.0)
        p_name = pred.get("display_name", p_class)
        
        col_ctx1, col_ctx2 = st.columns([5, 1])
        with col_ctx1:
            st.markdown(
                f"""
                <div class="leaf-context-banner">
                    <span class="context-tag">📸 {t('chat_context_badge')}</span>
                    <span style="color: #1B4D3E; font-weight: 600;">{p_name} ({p_conf:.1f}% confidence)</span>
                    <span style="color: #747E78; font-size: 0.88rem;">— Chat responses will incorporate this leaf diagnosis.</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_ctx2:
            if st.button("✖ Clear Context", help="Detach the scanned leaf prediction from chatbot queries"):
                st.session_state.current_prediction = None
                st.rerun()

    # Sidebar / Suggested Example Questions
    st.markdown(f"##### 💡 {t('example_questions_header')}")
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        if st.button(t("q1"), key="ex_q1", use_container_width=True):
            st.session_state.pending_chat_prompt = t("q1")
        if st.button(t("q2"), key="ex_q2", use_container_width=True):
            st.session_state.pending_chat_prompt = t("q2")
    with ex_col2:
        if st.button(t("q3"), key="ex_q3", use_container_width=True):
            st.session_state.pending_chat_prompt = t("q3")
        if st.button(t("q4"), key="ex_q4", use_container_width=True):
            st.session_state.pending_chat_prompt = t("q4")

    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)

    # Render Chat History
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 {t('sources_expander_title')} ({len(msg['sources'])} chunks)"):
                    for i, src in enumerate(msg["sources"]):
                        st.markdown(
                            f"""
                            <div class="source-citation-card">
                                <div class="source-title">📄 Source [{i+1}]: {src.get('source', 'Research Document')}</div>
                                <div style="color: #495057;">{src.get('text', '')[:280]}...</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # Chat Input Handling
    user_input = st.chat_input(t("chat_input_placeholder"))

    # If an example or disease integration button was pressed
    if st.session_state.pending_chat_prompt:
        user_input = st.session_state.pending_chat_prompt
        st.session_state.pending_chat_prompt = None

    if user_input:
        # Append User Message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate Grounded Response
        with st.chat_message("assistant"):
            with st.spinner("Searching Jimma & Ethiopian coffee research database..."):
                leaf_ctx = st.session_state.current_prediction if (st.session_state.current_prediction and st.session_state.current_prediction.get("success")) else None
                
                response_data = rag_bot.answer_question(
                    question=user_input,
                    leaf_context=leaf_ctx,
                    conversation_history=st.session_state.chat_messages,
                    lang=st.session_state.lang
                )

                answer_text = response_data.get("answer", "")
                sources = response_data.get("sources", [])

                st.markdown(answer_text)

                if sources:
                    with st.expander(f"📚 {t('sources_expander_title')} ({len(sources)} chunks)"):
                        for i, src in enumerate(sources):
                            st.markdown(
                                f"""
                                <div class="source-citation-card">
                                    <div class="source-title">📄 Source [{i+1}]: {src.get('source', 'Research Document')}</div>
                                    <div style="color: #495057;">{src.get('text', '')[:280]}...</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources
            })

    # Clear chat button
    if st.session_state.chat_messages:
        if st.button(f"🗑️ {t('clear_chat_button')}", key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()


# --- PAGE 4: ABOUT THE PROJECT ---
def render_about_page(rag_bot):
    st.markdown(f"## ℹ️ {t('nav_about')}")

    chunk_count = rag_bot.collection.count() if (rag_bot and rag_bot.collection) else "1,400+"

    st.markdown(
        f"""
        ### ☕ Ethiopian Coffee AI Assistant
        **Integrated Agricultural Decision Support System**

        This unified platform integrates two independently engineered AI systems into one cohesive solution designed for Ethiopian coffee growers, agronomists, and researchers.

        ---

        #### 1. Computer Vision Module (ResNet-18)
        - **Backbone Architecture:** PyTorch ResNet-18 with custom classification head (Dropout 0.3 -> Linear 512 -> ReLU -> Dropout 0.2 -> Linear 4).
        - **Target Diseases:** 
          1. **Coffee Leaf Rust** (*Hemileia vastatrix*)
          2. **Phoma Leaf Spot** (*Phoma tarda*)
          3. **Coffee Leaf Miner** (*Leucoptera coffeella*)
          4. **Healthy Leaf Foliage** (*Coffea arabica*)
        - **Inference Pipeline:** 224x224 RGB image normalization with real-time score distribution and `torch.inference_mode()`.

        #### 2. Retrieval-Augmented Generation (RAG) Chatbot
        - **Vector Database:** ChromaDB persistent storage containing **{chunk_count} verified research chunks** on Ethiopian coffee diseases, agronomic trials, and IPM protocols.
        - **Embedding Model:** `BAAI/bge-small-en-v1.5` (384-dimensional dense semantic representations).
        - **Grounding Architecture:** Real-time semantic retrieval with structured answer synthesis and direct citation of source literature.

        #### 3. Amharic Language Localization (አማርኛ)
        - **Bi-directional Translation:** Seamless translation of UI elements, domain-specific agricultural terminology, and query normalization.
        - **Ethiopic Script Support:** Native Unicode typography tailored for smallholder farmers across Ethiopia.

        ---

        #### ⚠️ Responsible Agricultural Advisory Disclaimer
        *Predictions and advice generated by this application are for educational and diagnostic decision-support purposes. Plant diseases can exhibit atypical symptoms depending on elevation, microclimate, and concurrent nutritional deficiencies. Always consult certified agricultural extension personnel before applying regulated chemicals or making major management interventions.*
        """
    )


# --- MAIN APPLICATION CONTROLLER ---
def main():
    render_brand_header()

    # Load AI engines
    vision_model, model_error = get_cached_vision_model()
    rag_bot, rag_error = get_cached_rag_chatbot()

    # Navigation options
    nav_options = [
        t("nav_home"),
        t("nav_detection"),
        t("nav_assistant"),
        t("nav_about")
    ]

    # Handle cross-page programmatic redirection (e.g. from buttons)
    selected_idx = 0
    if st.session_state.nav_target and st.session_state.nav_target in nav_options:
        selected_idx = nav_options.index(st.session_state.nav_target)
        st.session_state.nav_target = None

    selected_nav = st.sidebar.radio(
        label="📌 Navigation / ማውጫ",
        options=nav_options,
        index=selected_idx,
        key="main_navigation_radio"
    )

    # Sidebar System Health Status
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"#### ⚙️ {'System Status' if st.session_state.lang == 'en' else 'የስርዓቱ ሁኔታ'}")

    if vision_model is not None:
        st.sidebar.success(f"✅ Vision Model: ResNet-18 ({'Online' if st.session_state.lang == 'en' else 'ዝግጁ'})")
    else:
        st.sidebar.error("❌ Vision Model: Offline")

    if rag_bot and rag_bot.is_ready():
        c_count = rag_bot.collection.count()
        st.sidebar.success(f"✅ ChromaDB: {c_count} Chunks")
    else:
        st.sidebar.warning("⚠️ RAG Engine: Limited")

    st.sidebar.info(f"🌐 UI Language: {'English' if st.session_state.lang == 'en' else 'አማርኛ (Amharic)'}")

    # Dispatch to active page
    if selected_nav == t("nav_home"):
        render_home_page()
    elif selected_nav == t("nav_detection"):
        render_detection_page(vision_model, model_error)
    elif selected_nav == t("nav_assistant"):
        render_assistant_page(rag_bot, rag_error)
    elif selected_nav == t("nav_about"):
        render_about_page(rag_bot)

    # Footer
    st.markdown("<hr style='margin: 2.5rem 0 1rem 0; border: none; border-top: 1px solid #E8E2D9;'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="text-align: center; color: #747E78; font-size: 0.88rem; padding-bottom: 1.5rem;">
            🌱 <b>Ethiopian Coffee AI Assistant</b> | Jimma & National Coffee Research Integration | Built for Academic & Field Excellence
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
