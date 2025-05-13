import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import chainlit as cl # Import chainlit for its types

# Assuming your app structure allows these imports
# You might need to adjust paths or ensure __init__.py files are present
from app import (
    on_chat_start,
    _apply_chat_settings_to_state,
    execute_persona_tasks, 
    InsightFlowState,
    PersonaFactory, # Assuming PersonaFactory is accessible or can be mocked
    PERSONA_LLM_MAP, # Assuming this is accessible or can be mocked
    RAG_ENABLED_PERSONA_IDS # If this is defined globally in app.py
)
from utils.persona.base import PersonaReasoning # For creating mock persona objects

# Default RAG_ENABLED_PERSONA_IDS from app.py (or import it if it's moved to a config)
APP_RAG_ENABLED_PERSONA_IDS = ["analytical", "philosophical", "metaphorical"]

@pytest.fixture
def mock_cl_user_session():
    """Mocks chainlit.user_session with a dictionary-like store."""
    with patch('chainlit.user_session', new_callable=MagicMock) as mock_session:
        _session_store = {}
        def get_item(key, default=None):
            return _session_store.get(key, default)
        def set_item(key, value):
            _session_store[key] = value
        def contains_item(key):
            # Check if key is in _session_store, not mock_session itself
            return key in _session_store 

        mock_session.get = MagicMock(side_effect=get_item)
        mock_session.set = MagicMock(side_effect=set_item)
        # To make `key in cl.user_session` work as expected:
        mock_session.__contains__ = MagicMock(side_effect=contains_item)
        
        # Initialize with some common defaults if needed by other parts of app logic
        _session_store["id"] = "test_session_id"
        _session_store["persona_factory"] = PersonaFactory() # Real or mocked
        _session_store["embedding_model"] = MagicMock() # Assume it's initialized
        
        yield mock_session
        _session_store.clear() # Clean up after test

@pytest.fixture
def mock_persona_factory():
    factory = MagicMock(spec=PersonaFactory)
    # Setup mock personas that the factory can create
    mock_analytical_persona = MagicMock(spec=PersonaReasoning)
    mock_analytical_persona.id = "analytical"
    mock_analytical_persona.name = "Analytical Persona"
    mock_analytical_persona.expertise = "Logic and reason"
    mock_analytical_persona.role = "To analyze deeply"
    mock_analytical_persona.generate_perspective = AsyncMock(return_value="Analytical perspective")

    mock_philosophical_persona = MagicMock(spec=PersonaReasoning)
    mock_philosophical_persona.id = "philosophical"
    mock_philosophical_persona.name = "Philosophical Persona"
    mock_philosophical_persona.expertise = "Wisdom and ethics"
    mock_philosophical_persona.role = "To ponder deeply"
    mock_philosophical_persona.generate_perspective = AsyncMock(return_value="Philosophical perspective")
    
    mock_scientific_persona = MagicMock(spec=PersonaReasoning) # Non-RAG example
    mock_scientific_persona.id = "scientific"
    mock_scientific_persona.name = "Scientific Persona"
    mock_scientific_persona.expertise = "Empirical evidence"
    mock_scientific_persona.role = "To investigate phenomena"
    mock_scientific_persona.generate_perspective = AsyncMock(return_value="Scientific perspective")

    def create_persona_side_effect(persona_id, llm):
        if persona_id == "analytical": return mock_analytical_persona
        if persona_id == "philosophical": return mock_philosophical_persona
        if persona_id == "scientific": return mock_scientific_persona
        return MagicMock(spec=PersonaReasoning, id=persona_id, name=f"{persona_id.capitalize()} Persona", generate_perspective=AsyncMock(return_value=f"{persona_id} perspective"))

    factory.create_persona = MagicMock(side_effect=create_persona_side_effect)
    # Mock get_available_personas if _apply_chat_settings_to_state uses it directly
    factory.get_available_personas = MagicMock(return_value={"analytical": "Analytical", "philosophical": "Philosophical", "scientific": "Scientific"})
    factory.persona_configs = { # For on_chat_start to build switches
        "analytical": {"name": "Analytical Persona"},
        "philosophical": {"name": "Philosophical Persona"},
        "scientific": {"name": "Scientific Persona"}
    }
    return factory

@pytest.mark.asyncio
async def test_on_chat_start_adds_rag_toggle(mock_cl_user_session, mock_persona_factory):
    """Test that on_chat_start includes the 'enable_rag' Switch."""
    mock_cl_user_session.get.side_effect = lambda key, default=None: mock_persona_factory if key == "persona_factory" else (_session_store.get(key,default) if '_session_store' in globals() else default) # Ensure factory is returned for this test
    
    # Configure the mock for cl.ChatSettings
    mock_chat_settings_instance = AsyncMock() # This will be returned by cl.ChatSettings()
    mock_chat_settings_instance.send = AsyncMock() # Ensure the send method is an AsyncMock

    with patch('app.initialize_configurations') as mock_init_config, \
         patch('app.PersonaFactory', return_value=mock_persona_factory), \
         patch('chainlit.ChatSettings', return_value=mock_chat_settings_instance) as mock_chat_settings_class, \
         patch('chainlit.Message') as mock_cl_message, \
         patch('app.get_embedding_model', return_value=MagicMock()): # Mock embedding model loading
        
        await on_chat_start()

        mock_chat_settings_class.assert_called_once() # Check that cl.ChatSettings class was called
        # mock_chat_settings_instance.send.assert_called_once() # Check that the send method on the instance was called

        args, _ = mock_chat_settings_class.call_args
        inputs_list = args[0] # The first positional argument is `inputs`
        
        rag_toggle_present = any(widget.id == "enable_rag" and isinstance(widget, cl.Switch) for widget in inputs_list)
        assert rag_toggle_present, "'enable_rag' Switch not found in ChatSettings"
        
        rag_toggle_widget = next(widget for widget in inputs_list if widget.id == "enable_rag")
        assert rag_toggle_widget.initial is True, "RAG toggle should be ON by default"

@pytest.mark.asyncio
async def test_apply_chat_settings_reads_rag_toggle(mock_cl_user_session, mock_persona_factory):
    """Test that _apply_chat_settings_to_state correctly reads and sets the RAG toggle."""
    mock_cl_user_session.get.side_effect = [
        {"enable_rag": False, "selected_team": "none"},  # First call from _apply_chat_settings_to_state for chat_settings
        InsightFlowState(selected_personas=[]),         # For insight_flow_state
        mock_persona_factory,                          # For persona_factory
        False,                                         # Subsequent get for enable_rag from session
    ]
    # More robust: use a dict for side_effect or a more stateful mock for cl.user_session.get
    # This simple list relies on call order, which is fragile.
    # Let's refine using the session_store approach in the fixture itself for the general case
    # For this specific test, we can control the return values for chat_settings specifically.

    _session_store_for_test = {
        "insight_flow_state": InsightFlowState(selected_personas=[]),
        "persona_factory": mock_persona_factory,
        "chat_settings": {"enable_rag": False, "selected_team": "none"} # Simulate UI sending False
    }
    def mock_get_specific(key, default=None):
        if key == "chat_settings": return _session_store_for_test["chat_settings"]
        if key == "insight_flow_state": return _session_store_for_test["insight_flow_state"]
        if key == "persona_factory": return _session_store_for_test["persona_factory"]
        return _session_store_for_test.get(key, default)
    
    mock_cl_user_session.get = MagicMock(side_effect=mock_get_specific)

    await _apply_chat_settings_to_state()
    
    # Check that cl.user_session.set was called to update 'enable_rag'
    # Find the call to set 'enable_rag'
    set_rag_call = next((call for call in mock_cl_user_session.set.call_args_list if call[0][0] == 'enable_rag'), None)
    assert set_rag_call is not None, "cl.user_session.set was not called for 'enable_rag'"
    assert set_rag_call[0][1] is False, "'enable_rag' in session should be False"


@pytest.mark.asyncio
@patch('app.get_relevant_context_for_query', new_callable=AsyncMock) # Mock the RAG context function
async def test_execute_persona_tasks_rag_toggle_on(mock_get_context, mock_cl_user_session, mock_persona_factory):
    """Test execute_persona_tasks: RAG attempted when toggle is ON for RAG-enabled persona."""
    mock_cl_user_session.set("enable_rag", True)
    mock_cl_user_session.set("persona_factory", mock_persona_factory)
    mock_cl_user_session.set("embedding_model", MagicMock()) # Mocked embedding model
    mock_cl_user_session.set("progress_msg", AsyncMock(spec=cl.Message)) # Mock progress message
    mock_cl_user_session.set("completed_steps_log", [])

    # Mock PERSONA_LLM_MAP or ensure it's correctly populated for 'analytical'
    with patch.dict(PERSONA_LLM_MAP, {"analytical": MagicMock(spec=ChatOpenAI)}, clear=True):
        initial_state = InsightFlowState(
            query="test query for analytical",
            selected_personas=["analytical"], # RAG-enabled persona
            persona_responses={}
        )
        mock_get_context.return_value = "Retrieved RAG context."

        final_state = await execute_persona_tasks(initial_state)

        mock_get_context.assert_called_once_with("test query for analytical", "analytical", mock_cl_user_session.get("embedding_model"))
        # Check if the persona's generate_perspective was called with an augmented prompt
        analytical_persona_mock = mock_persona_factory.create_persona("analytical", None) # Get the mock
        call_args = analytical_persona_mock.generate_perspective.call_args[0][0]
        assert "Retrieved RAG context:" in call_args
        assert "User Query: test query for analytical" in call_args
        assert final_state["persona_responses"]["analytical"] == "Analytical perspective"

@pytest.mark.asyncio
@patch('app.get_relevant_context_for_query', new_callable=AsyncMock)
async def test_execute_persona_tasks_rag_toggle_off(mock_get_context, mock_cl_user_session, mock_persona_factory):
    """Test execute_persona_tasks: RAG NOT attempted when toggle is OFF."""
    mock_cl_user_session.set("enable_rag", False) # RAG is OFF
    mock_cl_user_session.set("persona_factory", mock_persona_factory)
    mock_cl_user_session.set("embedding_model", MagicMock())
    mock_cl_user_session.set("progress_msg", AsyncMock(spec=cl.Message))
    mock_cl_user_session.set("completed_steps_log", [])

    with patch.dict(PERSONA_LLM_MAP, {"analytical": MagicMock(spec=ChatOpenAI)}, clear=True):
        initial_state = InsightFlowState(
            query="test query for analytical rag off",
            selected_personas=["analytical"],
            persona_responses={}
        )
        
        final_state = await execute_persona_tasks(initial_state)

        mock_get_context.assert_not_called()
        # Check if the persona's generate_perspective was called with the original query
        analytical_persona_mock = mock_persona_factory.create_persona("analytical", None)
        call_args = analytical_persona_mock.generate_perspective.call_args[0][0]
        assert "Retrieved RAG context:" not in call_args # Original prompt structure without RAG context part
        assert "User Query: test query for analytical rag off" in call_args # Should be original query or a non-RAG augmented one
        assert "No specific context from your knowledge base was retrieved" in call_args # Or check for the non-RAG prompt
        assert final_state["persona_responses"]["analytical"] == "Analytical perspective"

@pytest.mark.asyncio
@patch('app.get_relevant_context_for_query', new_callable=AsyncMock)
async def test_execute_persona_tasks_rag_on_non_rag_persona(mock_get_context, mock_cl_user_session, mock_persona_factory):
    """Test execute_persona_tasks: RAG NOT attempted for non-RAG-enabled persona even if toggle is ON."""
    mock_cl_user_session.set("enable_rag", True) # RAG is ON globally
    mock_cl_user_session.set("persona_factory", mock_persona_factory)
    mock_cl_user_session.set("embedding_model", MagicMock())
    mock_cl_user_session.set("progress_msg", AsyncMock(spec=cl.Message))
    mock_cl_user_session.set("completed_steps_log", [])

    with patch.dict(PERSONA_LLM_MAP, {"scientific": MagicMock(spec=ChatOpenAI)}, clear=True):
        initial_state = InsightFlowState(
            query="test query for scientific",
            selected_personas=["scientific"], # NON-RAG-enabled persona
            persona_responses={}
        )
        
        final_state = await execute_persona_tasks(initial_state)

        mock_get_context.assert_not_called()
        scientific_persona_mock = mock_persona_factory.create_persona("scientific", None)
        call_args = scientific_persona_mock.generate_perspective.call_args[0][0]
        assert "Retrieved RAG context:" not in call_args
        assert call_args == "test query for scientific" # Original query passed directly
        assert final_state["persona_responses"]["scientific"] == "Scientific perspective"

# Need to import ChatOpenAI for the PERSONA_LLM_MAP patching to work if it's type hinted.
from langchain_openai import ChatOpenAI 