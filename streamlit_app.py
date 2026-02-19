import streamlit as st
import os
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Middletown RI AI", page_icon="🏛️")
st.title("🏛️ Middletown, RI AI Assistant")

# --- 2. GEMINI SETUP ---
# This looks for the key you put in Streamlit "Secrets"
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please add your GOOGLE_API_KEY to Streamlit Secrets!")

Settings.llm = Gemini(model="models/gemini-1.5-flash")

# --- 3. LOAD DATA ---
@st.cache_resource(show_spinner="Connecting to Town Records...")
def get_index():
    # Make sure this folder 'middletown_db' exists in your GitHub or is created by your crawler
    db = chromadb.PersistentClient(path="./middletown_db")
    chroma_collection = db.get_collection("middletown_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return load_index_from_storage(StorageContext.from_defaults(vector_store=vector_store))

try:
    index = get_index()
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_plus_context")
except Exception as e:
    st.warning("Database not found. Have you run the crawler yet?")
    st.stop()

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# New Input
if prompt := st.chat_input("Ask about Middletown..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.session_state.chat_engine.chat(prompt)
        st.markdown(response.response)
        st.session_state.messages.append({"role": "assistant", "content": response.response})
