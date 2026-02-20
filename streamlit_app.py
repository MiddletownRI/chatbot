import streamlit as st
import requests
from bs4 import BeautifulSoup
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Document, Settings

# --- 1. SETUP ---
st.set_page_config(page_title="Middletown RI AI", page_icon="🏛️")
st.title("🏛️ Middletown, RI AI Assistant")

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    Settings.llm = GoogleGenAI(model="models/gemini-1.5-flash", api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
else:
    st.error("Add GOOGLE_API_KEY to Secrets!")
    st.stop()

# --- 2. SIMPLE SCRAPER (No Playwright, No Loops) ---
def simple_scrape():
    try:
        url = "https://www.middletownri.gov"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Just grab the main text
        text = soup.get_text(separator=' ', strip=True)
        return [Document(text=text, metadata={"source": url})]
    except Exception as e:
        st.error(f"Scrape failed: {e}")
        return []

# --- 3. THE "STICKY" BRAIN ---
# Using session_state for EVERYTHING to stop the loops
if "index" not in st.session_state:
    st.session_state.index = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("Admin")
    if st.button("🚀 Load Town Data"):
        with st.spinner("Reading MiddletownRI.gov..."):
            docs = simple_scrape()
            if docs:
                st.session_state.index = VectorStoreIndex.from_documents(docs)
                st.success("Data Loaded!")
                st.rerun()

# --- 5. CHAT ---
if st.session_state.index:
    # Use the index from session state
    chat_engine = st.session_state.index.as_chat_engine(chat_mode="context")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about Middletown..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        response = chat_engine.chat(prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response.response)
        st.session_state.messages.append({"role": "assistant", "content": response.response})
else:
    st.info("👈 Click the button to load the website data into the AI.")
