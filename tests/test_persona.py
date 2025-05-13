import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessageChunk

# Assuming utils.persona is in the Python path
from utils.persona import PersonaReasoning, PersonaFactory
import inspect
print(f"DEBUG: PersonaFactory imported from: {inspect.getfile(PersonaFactory)}")
print(f"DEBUG: PersonaReasoning imported from: {inspect.getfile(PersonaReasoning)}")

@pytest.fixture
def mock_llm():
    """Fixture for a mocked LLM instance."""
    llm = MagicMock(spec=BaseChatModel)
    
    # Mock the behavior of astream
    async def mock_astream_behavior(messages):
        # Simulate streaming chunks
        yield AIMessageChunk(content="Hello, ")
        yield AIMessageChunk(content="world!")
        # Simulate an empty chunk which can happen
        yield AIMessageChunk(content="")
        yield AIMessageChunk(content=" How are you?")

    llm.astream = MagicMock(side_effect=mock_astream_behavior)
    return llm

@pytest.mark.asyncio
async def test_persona_reasoning_generate_perspective(mock_llm):
    """Test that PersonaReasoning.generate_perspective calls its LLM correctly and returns aggregated content."""
    persona_id = "test_persona"
    name = "Test Persona"
    system_prompt = "You are a test persona."
    
    reasoning = PersonaReasoning(persona_id, name, system_prompt, mock_llm)
    
    query = "What is the meaning of life?"
    expected_response = "Hello, world! How are you?"
    
    actual_response = await reasoning.generate_perspective(query)
    
    # Verify LLM call
    mock_llm.astream.assert_called_once()
    call_args = mock_llm.astream.call_args[0][0] # Get the first positional argument (messages list)
    
    assert len(call_args) == 2
    assert isinstance(call_args[0], SystemMessage)
    assert call_args[0].content == system_prompt
    assert isinstance(call_args[1], HumanMessage)
    assert call_args[1].content == query
    
    # Verify aggregated response
    assert actual_response == expected_response

def test_persona_factory_initialization():
    """Test PersonaFactory initialization and config loading."""
    factory = PersonaFactory()
    assert len(factory.persona_configs) > 0 # Check that some configs are loaded
    assert "analytical" in factory.persona_configs
    assert factory.persona_configs["analytical"]["name"] == "Analytical"

def test_persona_factory_create_persona_success(mock_llm):
    """Test successful creation of a PersonaReasoning instance."""
    factory = PersonaFactory()
    persona_id = "analytical"
    
    persona_instance = factory.create_persona(persona_id, mock_llm)
    
    assert persona_instance is not None
    assert isinstance(persona_instance, PersonaReasoning)
    assert persona_instance.persona_id == persona_id
    assert persona_instance.name == factory.persona_configs[persona_id]["name"]
    assert persona_instance.system_prompt == factory.persona_configs[persona_id]["system_prompt"]
    assert persona_instance.llm == mock_llm

def test_persona_factory_create_persona_invalid_id(mock_llm):
    """Test creating a persona with an invalid ID returns None."""
    factory = PersonaFactory()
    persona_instance = factory.create_persona("non_existent_persona", mock_llm)
    assert persona_instance is None

def test_persona_factory_create_persona_no_llm():
    """Test creating a persona without an LLM instance returns None."""
    factory = PersonaFactory()
    # We need a way to pass a 'None' LLM or ensure BaseChatModel type hint isn't violated
    # For now, let's assume the type hint means it must be a BaseChatModel.
    # The implementation checks `if config and llm_instance:`
    # So passing a non-BaseChatModel or None should ideally be handled by create_persona.
    # Let's test with None if the type hint allows, or by how create_persona handles it.
    # The implementation prints an error if llm_instance is None, and returns None.
    
    # Patch print to check for the error message if desired, but for now, just check None return
    with patch('utils.persona.base.print') as mock_print: # Patched print in the correct module
        persona_instance = factory.create_persona("analytical", None) # Pass None for LLM
        assert persona_instance is None
        mock_print.assert_any_call("DEBUG Error: LLM instance not provided for persona analytical")


def test_get_available_personas():
    """Test that get_available_personas returns the expected dictionary."""
    factory = PersonaFactory()
    available = factory.get_available_personas()
    assert isinstance(available, dict)
    assert "analytical" in available
    assert available["analytical"] == "Analytical"
    assert len(available) == len(factory.persona_configs) 