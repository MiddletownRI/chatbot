import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

def launch_cited_chat():
    # 1. Load the existing index
    db = chromadb.PersistentClient(path="./middletown_db")
    chroma_collection = db.get_collection("middletown_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = load_index_from_storage(StorageContext.from_defaults(vector_store=vector_store))

    # 2. Setup the engine
    chat_engine = index.as_chat_engine(chat_mode="condense_plus_context")

    print("\n🏛️ Middletown RI AI Assistant (with Sources)")
    print("Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Get the response object
        response = chat_engine.chat(user_input)
       
        # Print AI answer
        print(f"\nAI: {response.response}")

        # 3. Pull and print the unique sources found
        sources = set()
        for node in response.source_nodes:
            url = node.metadata.get("url", "Unknown Source")
            sources.add(url)
       
        if sources:
            print("\nSOURCES:")
            for s in sources:
                print(f"- {s}")
        print("-" * 30)

if __name__ == "__main__":
    launch_cited_chat()
