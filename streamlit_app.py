import streamlit as st
import asyncio
import os

# --- 1. SETUP & SETTINGS ---
st.set_page_config(page_title="Middletown RI AI", page_icon="🏛️")
st.title("🏛️ Middletown, RI AI Assistant")

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from crawl4ai import AsyncWebCrawler
from llama_index.core import VectorStoreIndex, Document, Settings

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    Settings.llm = GoogleGenAI(model="models/gemini-1.5-flash", api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
else:
    st.error("Missing GOOGLE_API_KEY in Secrets!")
    st.stop()

# --- 2. THE CRAWLER ---
async def crawl_town_site():
    async with AsyncWebCrawler() as crawler:
        # We'll crawl the main page and a couple of key sections to give it more "meat"
        result = await crawler.arun(url="https://www.middletownri.gov")
        if result.success:
            return [Document(text=result.markdown, metadata={"source": "Middletown Main Page"})]
        return []

# --- 3. SIDEBAR BUILDER ---
with st.sidebar:
    st.header("Admin")
    # If the index exists in RAM, show a 'Success' message
    if "index" in st.session_state:
        st.success("✅ Knowledge Base Loaded")
        if st.button("♻️ Re-build/Clear"):
            del st.session_state.index
            st.rerun()
    else:
        if st.button("🚀 Initial Build/Crawl"):
            with st.status("Crawling & Indexing...", expanded=True) as status:
                import nest_asyncio
                nest_asyncio.apply()
                
                # 1. Crawl
                docs = asyncio.run(crawl_town_site())
                
                if docs:
                    # 2. Index (Straight to RAM, no ChromaDB folder needed)
                    st.session_state.index = VectorStoreIndex.from_documents(docs)
                    status.update(label="Build Complete!", state="complete")
                    st.rerun()
                else:
                    st.error("Crawl failed. Please try again.")

# --- 4. CHAT INTERFACE ---
if "index" in st.session_state:
    # Build engine from the RAM-stored index
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = st.session_state.index.as_chat_engine(chat_mode="context")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): 
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about Middletown..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response = st.session_state.chat_engine.chat(prompt)
            st.markdown(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})
else:
    st.info("👈 Please click 'Initial Build' in the sidebar to load the town data into memory.")
