"""
Streamlit App - Asistente Turístico de Tenerife
"""

import streamlit as st
from src.conf import MODEL_CONFIG, PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from src.api_client import OpenAIClient
from src.data_loader import DataLoader
from src.vector_store import VectorStore
from src.rag_chain import RAGChain
from src.weather_service import WeatherService


# --- Page Config ---
st.set_page_config(
    page_title="Asistente Tenerife",
    page_icon="🏝️",
    layout="centered"
)


# --- Initialize Session State ---
@st.cache_resource
def init_rag_chain():
    """Initialize RAG chain (cached to avoid reloading on each interaction)."""
    client = OpenAIClient()
    loader = DataLoader(PDF_PATH)
    pages = loader.load()
    
    vector_store = VectorStore()
    vector_store.build_from_documents(pages)
    
    weather_service = WeatherService(simulated=True)
    rag = RAGChain(client, vector_store, weather_service=weather_service, max_history=5)
    
    return rag, loader.get_stats(), vector_store.get_chunk_stats()


# Initialize
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag" not in st.session_state:
    with st.spinner("🔄 Inicializando asistente..."):
        rag, doc_stats, chunk_stats = init_rag_chain()
        st.session_state.rag = rag
        st.session_state.doc_stats = doc_stats
        st.session_state.chunk_stats = chunk_stats


# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuración")
    
    st.subheader("Modelo")
    st.json(MODEL_CONFIG)
    
    st.subheader("RAG")
    st.write(f"- Chunk size: {CHUNK_SIZE}")
    st.write(f"- Chunk overlap: {CHUNK_OVERLAP}")
    st.write(f"- Total chunks: {st.session_state.chunk_stats['num_chunks']}")
    
    st.subheader("Documento")
    st.write(f"- Páginas: {st.session_state.doc_stats['num_pages']}")
    st.write(f"- Palabras: {st.session_state.doc_stats['total_words']:,}")
    
    st.divider()
    
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.session_state.rag.clear_history()
        st.rerun()


# --- Main Chat Interface ---
st.title("🏝️ Asistente Turístico de Tenerife")
st.caption("Pregunta sobre lugares, restaurantes, playas o el clima")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tool_called"):
            st.caption("🔧 Consultó pronóstico del tiempo")

# Chat input
if prompt := st.chat_input("Escribe tu pregunta..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            result = st.session_state.rag.query(prompt, k=5)
        
        st.markdown(result["answer"])
        
        if result.get("tool_called"):
            st.caption("🔧 Consultó pronóstico del tiempo")
    
    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "tool_called": result.get("tool_called", False)
    })