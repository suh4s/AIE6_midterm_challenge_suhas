"""
Utilities for Retrieval Augmented Generation (RAG) setup and operations.
"""
import os
import chainlit as cl
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Using HuggingFace for local, open-source model
from langchain_openai import OpenAIEmbeddings # Ensure this is imported
from langchain_qdrant import Qdrant
# from qdrant_client import QdrantClient # QdrantClient might be used for more direct/advanced Qdrant operations

# Recommended embedding model for good performance and local use.
# Ensure this model is downloaded or will be downloaded by sentence-transformers.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Qdrant recommends using FastEmbed for optimal performance with their DB.
# from langchain_community.embeddings import FastEmbedEmbeddings


def load_and_split_documents(persona_id: str, data_path: str = "data_sources"):
    """
    Loads all .txt files from the persona's data directory, splits them into
    chunks, and returns a list of Langchain Document objects.
    """
    persona_data_path = os.path.join(data_path, persona_id)
    if not os.path.isdir(persona_data_path):
        print(f"Warning: Data directory not found for persona {persona_id} at {persona_data_path}")
        return []

    # Using DirectoryLoader with UnstructuredFileLoader for .txt files
    # It's good at handling basic text. For more complex .txt or other formats,
    # you might need more specific loaders or pre-processing.
    loader = DirectoryLoader(
        persona_data_path,
        glob="**/*.txt", # Load all .txt files, including in subdirectories
        loader_cls=UnstructuredFileLoader,
        show_progress=True,
        use_multithreading=True,
        silent_errors=True # Suppress errors for files it can't load, log them instead if needed
    )
    
    try:
        loaded_documents = loader.load()
        if not loaded_documents:
            print(f"No documents found or loaded for persona {persona_id} from {persona_data_path}")
            return []
    except Exception as e:
        print(f"Error loading documents for persona {persona_id}: {e}")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust as needed
        chunk_overlap=150, # Adjust as needed
        length_function=len,
        is_separator_regex=False,
    )
    
    split_docs = text_splitter.split_documents(loaded_documents)
    print(f"Loaded and split {len(loaded_documents)} documents into {len(split_docs)} chunks for persona {persona_id}.")
    return split_docs


def get_embedding_model_instance(model_identifier: str, model_type: str = "hf"):
    """ 
    Initializes and returns an embedding model instance.
    model_type can be 'hf' for HuggingFace or 'openai'.
    """
    if model_type == "openai":
        print(f"Initializing OpenAI Embedding Model: {model_identifier}")
        return OpenAIEmbeddings(model=model_identifier)
    elif model_type == "hf":
        print(f"Initializing HuggingFace Embedding Model: {model_identifier}")
        # EMBEDDING_MODEL_NAME is still defined globally above, can be used as a fallback if needed
        # but app.py now passes the explicit model_identifier.
        return HuggingFaceEmbeddings(model_name=model_identifier) 
    else:
        raise ValueError(f"Unsupported embedding model_type: {model_type}")


async def get_or_create_persona_vector_store(persona_id: str, embedding_model):
    """Gets a vector store for a persona from the session, or creates and caches it."""
    vector_store_key = f"vector_store_{persona_id}"

    if cl.user_session.get(vector_store_key) is not None:
        print(f"Found existing vector store for {persona_id} in session.")
        return cl.user_session.get(vector_store_key)
    
    print(f"No existing vector store for {persona_id} in session. Creating new one...")
    documents = load_and_split_documents(persona_id)
    if not documents:
        print(f"No documents to create vector store for persona {persona_id}.")
        cl.user_session.set(vector_store_key, None) # Mark as attempted but failed
        return None

    try:
        # Qdrant.from_documents will handle creating the client and collection in-memory
        vector_store = await cl.make_async(Qdrant.from_documents)(
            documents,
            embedding_model,
            location=":memory:",  # Specifies in-memory Qdrant
            collection_name=f"{persona_id}_store_{cl.user_session.get('id', 'default_session')}", # Unique per session
            # distance_func="Cosine", # Qdrant default is Cosine, or use models.Distance.COSINE
            prefer_grpc=False # For local/in-memory, GRPC might not be necessary or set up
        )
        print(f"Successfully created vector store for {persona_id} with {len(documents)} chunks.")
        cl.user_session.set(vector_store_key, vector_store)
        return vector_store
    except Exception as e:
        print(f"Error creating Qdrant vector store for {persona_id}: {e}")
        cl.user_session.set(vector_store_key, None) # Mark as attempted but failed
        return None


async def get_relevant_context_for_query(query: str, persona_id: str, embedding_model) -> str:
    """
    Retrieves relevant context from the persona's vector store for a given query.
    Returns a string of concatenated context or an empty string if no context found.
    """
    vector_store = await get_or_create_persona_vector_store(persona_id, embedding_model)
    
    if not vector_store:
        print(f"No vector store available for {persona_id} to retrieve context.")
        return ""
        
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

    try:
        results = await cl.make_async(retriever.invoke)(query)
        if results:
            context = "\n\n---\n\n".join([doc.page_content for doc in results])
            # print(f"Retrieved context for {persona_id} query '{query}':\n{context[:500]}...") # Debug
            return context
        else:
            print(f"No relevant documents found for {persona_id} query: '{query}'")
            return ""
    except Exception as e:
        print(f"Error retrieving context for {persona_id} query '{query}': {e}")
        return "" 