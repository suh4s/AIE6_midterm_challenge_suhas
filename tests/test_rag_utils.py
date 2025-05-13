import pytest
import os
from unittest.mock import patch, MagicMock

# Make sure 'utils' is discoverable, or adjust path.
# This might require __init__.py in 'utils' and 'tests' and correct pythonpath.
from utils.rag_utils import load_and_split_documents, get_embedding_model, EMBEDDING_MODEL_NAME
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Tests for load_and_split_documents ---

def test_load_and_split_documents_no_directory(tmp_path):
    """Test behavior when the persona data directory does not exist."""
    persona_id = "non_existent_persona"
    result_docs = load_and_split_documents(persona_id, data_path=str(tmp_path))
    assert result_docs == []

def test_load_and_split_documents_no_txt_files(tmp_path):
    """Test behavior when directory exists but contains no .txt files."""
    persona_id = "empty_persona"
    persona_dir = tmp_path / persona_id
    persona_dir.mkdir()
    (persona_dir / "other_file.md").write_text("some markdown content")
    
    result_docs = load_and_split_documents(persona_id, data_path=str(tmp_path))
    assert result_docs == []

def test_load_and_split_documents_loads_and_splits_txt_files(tmp_path):
    """Test successful loading and splitting of .txt files."""
    persona_id = "test_persona"
    data_sources_path = tmp_path / "data_sources" # Simulate the data_sources structure
    data_sources_path.mkdir()
    persona_dir = data_sources_path / persona_id
    persona_dir.mkdir()

    # Create dummy .txt files
    (persona_dir / "doc1.txt").write_text("This is the first document. It has some text.")
    (persona_dir / "doc2.txt").write_text("Another document here with more words to ensure splitting might occur if long enough.")
    
    # Mocking DirectoryLoader.load() to control what it returns,
    # as testing the actual loader behavior deeply is out of scope for this unit test.
    # We are more interested in the interaction with text_splitter.
    # However, for this test, let's allow it to run to verify basic integration.
    # For more complex scenarios or to avoid actual file loading, mocking loader.load() would be better.

    # For simplicity, we'll assume RecursiveCharacterTextSplitter works as expected.
    # We are mainly testing that documents are loaded and passed to the splitter.
    
    split_docs = load_and_split_documents(persona_id, data_path=str(data_sources_path))
    
    assert len(split_docs) > 0 # Expecting at least one chunk per document if short, or more if split
    assert isinstance(split_docs[0], Document)
    
    # Check if content from original docs is present (simplified check)
    content_doc1_present = any("first document" in doc.page_content for doc in split_docs)
    content_doc2_present = any("Another document" in doc.page_content for doc in split_docs)
    assert content_doc1_present or content_doc2_present # At least one should be found if files are small

    # A more robust test would mock text_splitter.split_documents and verify it's called with loaded docs.


def test_load_and_split_documents_uses_correct_loader_and_splitter_params():
    """Test that DirectoryLoader and RecursiveCharacterTextSplitter are called with expected parameters."""
    persona_id = "params_test_persona"
    data_path = "dummy_data_path"
    dummy_persona_path = os.path.join(data_path, persona_id)

    # Mock os.path.isdir to simulate directory existence
    with patch('os.path.isdir', return_value=True):
        # Mock DirectoryLoader
        mock_doc_instance = Document(page_content="Test content from loader.")
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [mock_doc_instance] # Simulate loader returning one doc
        
        with patch('utils.rag_utils.DirectoryLoader', return_value=mock_loader_instance) as mock_directory_loader:
            # Mock RecursiveCharacterTextSplitter
            mock_splitter_instance = MagicMock()
            mock_splitter_instance.split_documents.return_value = [Document(page_content="Split chunk 1")] # Simulate splitter returning one chunk
            
            with patch('utils.rag_utils.RecursiveCharacterTextSplitter', return_value=mock_splitter_instance) as mock_text_splitter:
                
                load_and_split_documents(persona_id, data_path=data_path)

                # Assert DirectoryLoader was called correctly
                mock_directory_loader.assert_called_once_with(
                    dummy_persona_path,
                    glob="**/*.txt",
                    loader_cls=UnstructuredFileLoader,
                    show_progress=True,
                    use_multithreading=True,
                    silent_errors=True
                )
                mock_loader_instance.load.assert_called_once()

                # Assert RecursiveCharacterTextSplitter was called correctly
                mock_text_splitter.assert_called_once_with(
                    chunk_size=1000,
                    chunk_overlap=150,
                    length_function=len,
                    is_separator_regex=False
                )
                mock_splitter_instance.split_documents.assert_called_once_with([mock_doc_instance])


# --- Tests for get_embedding_model ---

def test_get_embedding_model_default():
    """Test that get_embedding_model returns a HuggingFaceEmbeddings instance with the default model."""
    # Patching the HuggingFaceEmbeddings constructor to avoid actual model loading/download
    with patch('utils.rag_utils.HuggingFaceEmbeddings') as mock_hf_embeddings:
        mock_instance = MagicMock(spec=HuggingFaceEmbeddings)
        mock_hf_embeddings.return_value = mock_instance
        
        embedding_model = get_embedding_model()
        
        mock_hf_embeddings.assert_called_once_with(model_name=EMBEDDING_MODEL_NAME)
        assert embedding_model == mock_instance

def test_get_embedding_model_custom_name():
    """Test get_embedding_model with a custom model name."""
    custom_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    with patch('utils.rag_utils.HuggingFaceEmbeddings') as mock_hf_embeddings:
        mock_instance = MagicMock(spec=HuggingFaceEmbeddings)
        mock_hf_embeddings.return_value = mock_instance

        embedding_model = get_embedding_model(model_name=custom_model)
        
        mock_hf_embeddings.assert_called_once_with(model_name=custom_model)
        assert embedding_model == mock_instance