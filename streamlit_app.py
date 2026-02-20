# --- 0. CRITICAL DATABASE FIX (MUST BE FIRST) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import asyncio
import os

# --- 1. HANDLE BROWSER INSTALLATION ---
if "playwright_installed" not in st.session_state:
    with st.spinner("Preparing browser engine..."):
        os.system("playwright install chromium")
        st.session_state.playwright_installed = True

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from crawl4ai import AsyncWebCrawler
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- 2. CONFIGURATION ---
st.set_page_config(page_title="Middletown RI AI", page_icon="🏛️")
st.title("🏛️ Middletown, RI AI Assistant")

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key 
    
    # The "Chef" (Gemini 2.0 Flash is state-of-the-art for 2026)
    Settings.llm = GoogleGenAI(
        model="models/gemini-2.0-flash", 
        api_key=api_key
    )

    # The "Cataloguer" (HuggingFace avoids the Google ClientError)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
else:
    st.error("Missing GOOGLE_API_KEY in Streamlit Secrets!")
    st.stop()

DB_PATH = "./middletown_db"

# --- 3. THE CRAWLER ---
async def crawl_town_site():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url="https://www.middletownri.gov")
        if result.success:
            return [Document(text=result.markdown, metadata={"source": "middletownri.gov"})]
        return []
        
# --- 4. SIDEBAR & BUILD LOGIC ---
with st.sidebar:
    st.header("Admin")
    if st.button("🚀 Initial Build/Crawl"):
        with st.status("Crawling MiddletownRI.gov...", expanded=True) as status:
            try:
                import nest_asyncio
                nest_asyncio.apply()
                docs = asyncio.run(crawl_town_site())
            except:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                docs = loop.run_until_complete(crawl_town_site())

            status.write("Creating vector database...")
            
            db = chromadb.PersistentClient(path=DB_PATH)
            chroma_collection = db.get_or_create_collection("middletown_docs")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Builds index from crawled documents
            index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
            index.storage_context.persist(persist_dir=DB_PATH)
            status.update(label="Build Complete!", state="complete")
            st.rerun()

# --- 5. CHAT LOGIC ---
if os.path.exists(DB_PATH):
    try:
        # 1. Reconnect to the Chroma database
        db = chromadb.PersistentClient(path=DB_PATH)
        chroma_collection = db.get_collection("middletown_docs")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # 2. Rebuild the storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=DB_PATH)
        
        # 3. Load the index (The safety check happens here)
        index = load_index_from_storage(storage_context)
        chat_engine = index.as_chat_engine(chat_mode="context")
        
        # --- Standard Streamlit Chat UI ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about Middletown..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                response = chat_engine.chat(prompt)
                st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
                
    except Exception as e:
        # If the files are missing or the DB is corrupted, show the info message instead of crashing
        st.info("👈 Data is not yet fully indexed. Please click 'Initial Build' in the sidebar.")
else:
    st.info("👈 Please click the 'Initial Build' button in the sidebar to load the town data.")
