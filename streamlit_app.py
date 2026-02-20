import streamlit as st
import asyncio
import os
if "playwright_installed" not in st.session_state:
    with st.spinner("Preparing browser engine..."):
        os.system("playwright install chromium")
        st.session_state.playwright_installed = True
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from crawl4ai import AsyncWebCrawler
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Middletown RI AI", page_icon="🏛️")
st.title("🏛️ Middletown, RI AI Assistant")

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key # Keeps other tools happy
    
    # 1. The "Chef" (Generates responses)
    Settings.llm = GoogleGenAI(
        model="models/gemini-3-flash", 
        api_key=api_key
    )
    
    # 2. The "Cataloguer" (Turns website text into searchable data)
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="models/text-embedding-004", 
        api_key=api_key
    )
else:
    st.error("Missing GOOGLE_API_KEY in Streamlit Secrets!")
    st.stop()

DB_PATH = "./middletown_db"

# --- 2. THE CRAWLER (The "Data Gatherer") ---
async def crawl_town_site():
    async with AsyncWebCrawler() as crawler:
        # Start with the main page
        result = await crawler.arun(url="https://www.middletownri.gov")
        if result.success:
            return [Document(text=result.markdown, metadata={"source": "middletownri.gov"})]
        return []

# --- 3. SIDEBAR CONTROLS ---
# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Admin")
    if st.button("🚀 Initial Build/Crawl"):
        with st.status("Crawling MiddletownRI.gov...", expanded=True) as status:
            # FIX: Use a helper to run the async crawler safely
            try:
                import nest_asyncio
                nest_asyncio.apply()
                docs = asyncio.run(crawl_town_site())
            except:
                # Fallback for some environments
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                docs = loop.run_until_complete(crawl_town_site())

            status.write("Creating vector database...")
            
            db = chromadb.PersistentClient(path=DB_PATH)
            chroma_collection = db.get_or_create_collection("middletown_docs")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
            status.update(label="Build Complete!", state="complete")
            st.rerun()

# --- 4. CHAT LOGIC ---
if os.path.exists(DB_PATH):
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_collection("middletown_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = load_index_from_storage(StorageContext.from_defaults(vector_store=vector_store))
    chat_engine = index.as_chat_engine(chat_mode="context")
    
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
else:
    st.info("👈 Please click the 'Initial Build' button in the sidebar to load the town data.")
