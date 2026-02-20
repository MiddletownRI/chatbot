__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import asyncio
import os
import chromadb
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from crawl4ai import AsyncWebCrawler
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore

# 1. GLOBAL SETTINGS
st.set_page_config(page_title="Middletown RI AI", page_icon="🏛️")
st.title("🏛️ Middletown, RI AI Assistant")

DB_PATH = "./middletown_db"

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    Settings.llm = GoogleGenAI(model="models/gemini-1.5-flash", api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
else:
    st.error("Missing GOOGLE_API_KEY in Secrets!")
    st.stop()

# 2. CACHED LOADING (This prevents the loop)
@st.cache_resource(show_spinner="Connecting to Knowledge Base...")
def load_cached_index():
    if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
        return None
    try:
        db = chromadb.PersistentClient(path=DB_PATH)
        chroma_collection = db.get_collection("middletown_docs")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=DB_PATH)
        return load_index_from_storage(storage_context)
    except Exception:
        return None

# 3. SIDEBAR (The Builder)
with st.sidebar:
    st.header("Admin")
    if st.button("🚀 Initial Build/Crawl"):
        with st.status("Building...", expanded=True) as status:
            # Crawl
            import nest_asyncio
            nest_asyncio.apply()
            async def do_crawl():
                async with AsyncWebCrawler() as crawler:
                    res = await crawler.arun(url="https://www.middletownri.gov")
                    return [Document(text=res.markdown)] if res.success else []
            docs = asyncio.run(do_crawl())
            
            # Index & Persist
            db = chromadb.PersistentClient(path=DB_PATH)
            chroma_collection = db.get_or_create_collection("middletown_docs")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
            index.storage_context.persist(persist_dir=DB_PATH)
            
            # Clear cache so the app 'sees' the new data
            st.cache_resource.clear()
            status.update(label="Build Complete!", state="complete")
            st.rerun()

# 4. CHAT INTERFACE
index = load_cached_index()

if index:
    # Initialize chat engine once in session state
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="context")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about Middletown..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # Use session_state engine to keep conversation history
        response = st.session_state.chat_engine.chat(prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response.response)
        st.session_state.messages.append({"role": "assistant", "content": response.response})
else:
    st.info("👈 Please click 'Initial Build' in the sidebar to start.")
