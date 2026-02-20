# streamlit_app.py
"""
RAG_AI — Streamlit UI
Replaces the old HTML/JS frontend with a Streamlit app.
Run:  streamlit run streamlit_app.py
"""

import os
import sys
import shutil
import streamlit as st

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.vector_store import VectorStore
from database.sql_store import SQLStore
from retrieval.retriever import Retriever
from llm.chain import (
    RAGChain, GROQ_MODELS, OLLAMA_MODELS,
    DEFAULT_LLM_PROVIDER, OLLAMA_MODEL, GROQ_MODEL,
)
import llm.chain as chain_module

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─────────────────────────────────────
# Cached singleton resources
# ─────────────────────────────────────
@st.cache_resource
def init_stores():
    vs = VectorStore(index_path="database/faiss.index")
    ss = SQLStore()
    ret = Retriever(vector_store=vs)
    rag = RAGChain(retriever=ret)
    return vs, ss, ret, rag


vector_store, sql_store, retriever, rag_chain = init_stores()


# ─────────────────────────────────────
# Page config
# ─────────────────────────────────────
st.set_page_config(
    page_title="RAG_AI — Intelligent Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar branding */
    [data-testid="stSidebar"] { background: #1a1b26; }
    [data-testid="stSidebar"] .stRadio label { font-size: 15px; }

    /* Chat bubbles */
    .user-msg {
        background: linear-gradient(135deg, #7c5cfc, #9178ff);
        color: #fff; padding: 12px 18px; border-radius: 14px 14px 4px 14px;
        margin: 8px 0; max-width: 75%; margin-left: auto; text-align: right;
    }
    .ai-msg {
        background: #1e1f32; border: 1px solid #2e3044;
        color: #e1e2e8; padding: 16px 20px; border-radius: 4px 14px 14px 14px;
        margin: 8px 0; max-width: 85%; line-height: 1.7;
    }
    .provider-badge {
        display: inline-block; font-size: 11px; font-weight: 600;
        padding: 2px 10px; border-radius: 20px;
        background: #24253a; color: #9ca0b0; border: 1px solid #2e3044;
        margin-top: 6px;
    }
    .source-chip {
        display: inline-block; background: #24253a; border: 1px solid #2e3044;
        border-radius: 6px; padding: 4px 8px; margin: 2px 3px; font-size: 12px;
        color: #9ca0b0;
    }
    .doc-card {
        background: #1e1f32; border: 1px solid #2e3044; border-radius: 12px;
        padding: 16px; margin-bottom: 10px;
    }
    .doc-type-badge {
        display: inline-block; font-size: 11px; font-weight: 600;
        text-transform: uppercase; padding: 3px 10px; border-radius: 20px;
        background: rgba(124,92,252,0.25); color: #9178ff;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ─────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 RAG_AI")
    st.caption("Intelligent Document Assistant")
    st.divider()

    page = st.radio(
        "Navigation",
        ["💬 Ask Question", "📤 Upload Document", "📚 Documents", "⚙️ Settings"],
        label_visibility="collapsed",
    )


# ═════════════════════════════════════
# 💬 ASK QUESTION
# ═════════════════════════════════════
if page == "💬 Ask Question":
    st.header("Ask a Question")
    st.caption("Query your ingested documents using AI-powered retrieval")

    # --- Filters row ---
    col1, col2, col3 = st.columns(3)
    with col1:
        doc_type = st.selectbox(
            "Document Type",
            ["Auto-detect", "pdf", "html", "json", "email", "image", "audio"],
            index=0,
        )
    with col2:
        docs = sql_store.get_all_documents()
        sources = sorted(set(d["source"] for d in docs if d.get("source")))
        source_options = ["All Documents"] + sources
        source_filter = st.selectbox("Source Filter", source_options, index=0)
    with col3:
        provider = st.selectbox(
            "LLM Provider",
            ["Auto", "ollama", "groq"],
            index=0,
        )

    # Model selector (appears when specific provider chosen)
    model_name = None
    if provider == "ollama":
        model_name = st.selectbox("Ollama Model", OLLAMA_MODELS, index=0)
    elif provider == "groq":
        model_name = st.selectbox("Groq Model", GROQ_MODELS, index=0)

    st.divider()

    # --- Chat history ---
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(msg["text"])
        else:
            with st.chat_message("assistant", avatar="🧠"):
                st.write(msg["text"])
                if msg.get("provider"):
                    st.caption(f"Provider: {msg['provider']}")
                if msg.get("sources"):
                    with st.expander(f"📎 Sources ({len(msg['sources'])})"):
                        for src in msg["sources"]:
                            st.markdown(f"```\n{src.get('text', '')}\n```")

    # --- Chat input ---
    question = st.chat_input("Type your question here...")

    if question:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "text": question})
        with st.chat_message("user", avatar="👤"):
            st.write(question)

        # Query
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Thinking..."):
                try:
                    result = rag_chain.query(
                        question=question,
                        top_k=5,
                        doc_type=doc_type if doc_type != "Auto-detect" else None,
                        source_filter=source_filter if source_filter != "All Documents" else None,
                        provider=provider if provider != "Auto" else None,
                        model_name=model_name,
                    )
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    provider_used = result.get("provider_used", "unknown")

                    st.write(answer)
                    st.caption(f"Provider: {provider_used}")

                    if sources:
                        with st.expander(f"📎 Sources ({len(sources)})"):
                            for src in sources:
                                st.markdown(f"```\n{src.get('text', '')}\n```")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "text": answer,
                        "sources": sources,
                        "provider": provider_used,
                    })

                except Exception as e:
                    error_msg = f"⚠️ Error: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", "text": error_msg,
                    })

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


# ═════════════════════════════════════
# 📤 UPLOAD DOCUMENT
# ═════════════════════════════════════
elif page == "📤 Upload Document":
    st.header("Upload Document")
    st.caption("Upload files to ingest into the RAG pipeline")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "html", "json", "eml", "png", "jpg", "jpeg", "wav", "mp3"],
        help="Supported: PDF, HTML, JSON, EML, Images (PNG, JPG), Audio (WAV, MP3)",
    )

    if uploaded_file is not None:
        st.info(f"📄 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

        if st.button("🚀 Ingest Document", type="primary"):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    # Save file
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Run ingestion
                    from main import ingest
                    ingest(file_path, vector_store=vector_store, sql_store=sql_store)

                    st.success(f"✅ **{uploaded_file.name}** ingested successfully!")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")

    st.divider()
    st.subheader("Supported File Types")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("📄 **PDF** — Documents, reports")
        st.markdown("🌐 **HTML** — Web pages")
    with col2:
        st.markdown("📋 **JSON** — Structured data")
        st.markdown("📧 **EML** — Email messages")
    with col3:
        st.markdown("🖼️ **PNG/JPG** — Images (OCR)")
        st.markdown("🎵 **WAV/MP3** — Audio files")


# ═════════════════════════════════════
# 📚 DOCUMENTS
# ═════════════════════════════════════
elif page == "📚 Documents":
    st.header("Ingested Documents")
    st.caption("All documents currently in the knowledge base")

    docs = sql_store.get_all_documents()

    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col_b:
        st.caption(f"{len(docs)} document{'s' if len(docs) != 1 else ''}")

    if not docs:
        st.info("No documents ingested yet. Go to **Upload Document** to get started!")
    else:
        # Group by unique source
        seen = set()
        unique_docs = []
        for d in docs:
            if d["source"] not in seen:
                seen.add(d["source"])
                unique_docs.append(d)

        for doc in unique_docs:
            source = doc.get("source", "Unknown")
            doc_type = doc.get("doc_type", "unknown")

            # File icon
            ext = source.rsplit(".", 1)[-1].lower() if "." in source else ""
            icons = {"pdf": "📄", "html": "🌐", "json": "📋", "eml": "📧",
                     "png": "🖼️", "jpg": "🖼️", "jpeg": "🖼️", "wav": "🎵", "mp3": "🎵"}
            icon = icons.get(ext, "📁")

            with st.container():
                col1, col2, col3 = st.columns([0.5, 4, 1])
                with col1:
                    st.markdown(f"<span style='font-size:28px'>{icon}</span>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**{source}**")
                    st.caption(f"Type: {doc_type.upper()}")
                with col3:
                    if st.button("🗑️ Delete", key=f"del_{source}", type="secondary"):
                        with st.spinner(f"Deleting {source}..."):
                            try:
                                chunks_removed = vector_store.delete_by_source(source)
                                sql_store.delete_documents_by_source(source)
                                upload_path = os.path.join(UPLOAD_DIR, source)
                                if os.path.exists(upload_path):
                                    os.remove(upload_path)
                                st.success(f"Deleted {source}: {chunks_removed} chunks removed.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete: {e}")
                st.divider()


# ═════════════════════════════════════
# ⚙️ SETTINGS
# ═════════════════════════════════════
elif page == "⚙️ Settings":
    st.header("Settings")
    st.caption("Configure LLM providers and API keys")

    # --- Provider Status ---
    st.subheader("🤖 LLM Providers")
    groq_ok = bool(chain_module.GROQ_API_KEY and chain_module.GROQ_API_KEY != "your_groq_api_key_here")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🏠 Ollama (Local)**")
        st.success("● Available")
        st.caption(f"Default model: `{chain_module.OLLAMA_MODEL}`")
        st.caption(f"Models: {', '.join(chain_module.OLLAMA_MODELS)}")
    with col2:
        st.markdown("**☁️ Groq (Cloud)**")
        if groq_ok:
            st.success("● Connected")
        else:
            st.warning("○ No API key configured")
        st.caption(f"Default model: `{chain_module.GROQ_MODEL}`")
        st.caption(f"Models: {', '.join(chain_module.GROQ_MODELS)}")

    st.divider()

    # --- Groq API Key ---
    st.subheader("⚡ Groq API Key")
    st.caption("Get your free API key at [console.groq.com/keys](https://console.groq.com/keys)")

    groq_key = st.text_input(
        "API Key",
        type="password",
        placeholder="gsk_xxxxxxxxxxxxxxxx",
        label_visibility="collapsed",
    )

    if st.button("💾 Save API Key", type="primary"):
        if not groq_key.strip():
            st.warning("Please enter an API key.")
        else:
            # Update runtime
            chain_module.GROQ_API_KEY = groq_key.strip()

            # Persist to .env
            env_path = os.path.join(PROJECT_ROOT, ".env")
            try:
                if os.path.exists(env_path):
                    with open(env_path, "r") as f:
                        lines = f.readlines()
                    with open(env_path, "w") as f:
                        found = False
                        for line in lines:
                            if line.startswith("GROQ_API_KEY="):
                                f.write(f"GROQ_API_KEY={groq_key.strip()}\n")
                                found = True
                            else:
                                f.write(line)
                        if not found:
                            f.write(f"\nGROQ_API_KEY={groq_key.strip()}\n")
                else:
                    with open(env_path, "w") as f:
                        f.write(f"GROQ_API_KEY={groq_key.strip()}\n")

                st.success("✅ Groq API key saved!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")

    st.divider()

    # --- Default Provider ---
    st.subheader("🔧 Default Provider")
    current = chain_module.DEFAULT_LLM_PROVIDER
    new_provider = st.radio(
        "Choose default LLM provider",
        ["ollama", "groq"],
        index=0 if current == "ollama" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    if new_provider != current:
        chain_module.DEFAULT_LLM_PROVIDER = new_provider
        st.toast(f"Default provider set to **{new_provider}**")
