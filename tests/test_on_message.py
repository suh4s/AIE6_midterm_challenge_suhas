import pytest
from unittest.mock import AsyncMock, MagicMock, patch, ANY
import chainlit as cl
import sys
import app
from langchain_openai import ChatOpenAI
import langchain_openai
import inspect
from langchain_core.messages import AIMessageChunk

# Functions/classes to be tested (from app.py)
# We need to ensure app.py can be imported or we define these structures similarly
# For now, assuming app.on_message and app.InsightFlowState are accessible for mocking/patching
# from app import on_message, InsightFlowState # Ideal import

# Let's use the state definition from state.py for consistency
from insight_state import InsightFlowState
from utils.persona import PersonaFactory, PersonaReasoning
from langchain_core.messages import AIMessageChunk

# Placeholder for the actual on_message if it were in a separate module
# For now, we'll patch where it's defined (globally if app.py is run directly)
# This path might need adjustment based on how Chainlit runs the app
APP_PY_PATH = "app"

# Use the actual InsightFlowState for type consistency if needed, but primarily for session storage key
from insight_state import InsightFlowState

# Helper function to set up mock_cl behavior for these tests
def setup_mock_cl_session_get(mock_cl, direct_mode_value, quick_mode_value=False, initial_state_dict=None):
    # This function will be called by cl.user_session.get(key)
    
    default_state = {
        "query": "", "selected_personas": [], "persona_responses": {},
        "synthesized_response": None, "visualization_code": None,
        "visualization_image_url": None, "current_step_name": "awaiting_query",
        "error_message": None, "panel_type": "research"
    }
    current_state = initial_state_dict if initial_state_dict is not None else default_state

    def side_effect_func(key, default=None): # Added default to match real signature
        if key == "direct_mode":
            return direct_mode_value
        elif key == "quick_mode":
            return quick_mode_value
        elif key == "insight_flow_state":
            return current_state
        # Add other session variables if they are retrieved directly in on_message or invoke_langgraph
        elif key == "persona_factory": # From test_invoke_langgraph_orchestrates_persona_execution
            return session_data.get(key, MagicMock(spec=PersonaFactory)) if 'session_data' in globals() and session_data else MagicMock(spec=PersonaFactory)
        elif key == "id": # From test_invoke_langgraph_orchestrates_persona_execution
             return session_data.get(key, "default_thread_id") if 'session_data' in globals() and session_data else "default_thread_id"
        return MagicMock() # Default for other keys
    mock_cl.user_session.get.side_effect = side_effect_func
    return current_state # Return the state that will be used, for convenience in tests

@pytest.mark.asyncio
async def test_on_message_direct_mode_on_calls_direct_llm(mock_cl): # Use mock_cl from conftest
    """Test that on_message calls the direct LLM if direct_mode is True."""
    mock_incoming_message = MagicMock(spec=cl.Message)
    mock_incoming_message.content = "Test query for direct mode"

    setup_mock_cl_session_get(mock_cl, direct_mode_value=True)
    # Ensure cl.Message().send() is an AsyncMock
    mock_cl.Message.return_value.send = AsyncMock()
    
    # Patch app.cl directly for on_message handler context
    with patch(f'{APP_PY_PATH}.cl', new=mock_cl):
        with patch(f'{APP_PY_PATH}.invoke_direct_llm', new_callable=AsyncMock) as mock_direct_call:
            with patch(f'{APP_PY_PATH}.invoke_langgraph', new_callable=AsyncMock) as mock_graph_call:
                from app import on_message # Import under patch context
                await on_message(mock_incoming_message)

    mock_direct_call.assert_called_once_with(mock_incoming_message.content)
    mock_graph_call.assert_not_called()
    mock_cl.user_session.get.assert_any_call("direct_mode")

@pytest.mark.asyncio
async def test_on_message_direct_mode_off_calls_langgraph(mock_cl): # Use mock_cl from conftest
    """Test that on_message calls LangGraph if direct_mode is False and progress messages are handled."""
    mock_incoming_message = MagicMock(spec=cl.Message)
    mock_incoming_message.content = "Test query for graph mode"

    expected_initial_state = setup_mock_cl_session_get(mock_cl, direct_mode_value=False)
    
    # --- Mock cl.Message for progress updates --- #
    # We need to capture the instance of the progress message to check its methods.
    mock_progress_message_instance = AsyncMock(spec=cl.Message) # Mock the instance
    mock_progress_message_instance.send = AsyncMock()
    mock_progress_message_instance.stream_token = AsyncMock()
    mock_progress_message_instance.update = AsyncMock()

    # Configure mock_cl.Message constructor to return our specific instance when content is ""
    # For other cl.Message calls (e.g. in present_results), it can return a default MagicMock
    default_mock_message_instance = AsyncMock(spec=cl.Message, send=AsyncMock())
    def message_side_effect(*args, **kwargs):
        if kwargs.get("content") == "": # This is how progress_msg is initialized
            return mock_progress_message_instance
        return default_mock_message_instance # For any other messages created
    mock_cl.Message.side_effect = message_side_effect
    mock_cl.Message.reset_mock() # Reset call count for the Message class itself from previous tests using mock_cl

    # To store the actual progress message instance passed to the callback handler
    passed_progress_msg_to_callback = None

    # --- Mock InsightFlowCallbackHandler --- #    
    # We need to capture the arguments passed to its constructor, especially the progress_message.
    # This class will be instantiated by the side_effect of our main mock.
    class MockInsightFlowCallbackHandler(app.InsightFlowCallbackHandler):
        def __init__(self, progress_message: cl.Message):
            nonlocal passed_progress_msg_to_callback
            passed_progress_msg_to_callback = progress_message
            # super().__init__(progress_message) # We don't need to call super for this mock.
                                            # The instance itself can be a simple object for type checking
                                            # and argument capture verification.
            self.progress_message = progress_message # Store it for potential inspection if needed

    # This function will be the side_effect for the MagicMock replacing app.InsightFlowCallbackHandler
    # It ensures our local MockInsightFlowCallbackHandler is created, capturing the args.
    def mock_ifch_constructor_side_effect(progress_message):
        return MockInsightFlowCallbackHandler(progress_message=progress_message)

    # Patch app.cl (used by on_message and invoke_langgraph) and other app internals
    with patch(f'{APP_PY_PATH}.cl', new=mock_cl): # Patches app.cl
        with patch(f'{APP_PY_PATH}.invoke_direct_llm', new_callable=AsyncMock) as mock_direct_call:
            # We want the real invoke_langgraph to run, so we patch what IT calls:
            with patch(f'{APP_PY_PATH}.insight_flow_graph.ainvoke', new_callable=AsyncMock) as mock_graph_ainvoke:
                # Patch the CallbackHandler class with a MagicMock.
                # Its side_effect will use our local MockInsightFlowCallbackHandler for instantiation.
                with patch(f'{APP_PY_PATH}.InsightFlowCallbackHandler') as mock_actual_ifch_class_constructor:
                    mock_actual_ifch_class_constructor.side_effect = mock_ifch_constructor_side_effect
                    from app import on_message # Import under patch context
                    await on_message(mock_incoming_message)

    # --- Assertions --- #
    mock_direct_call.assert_not_called()
    
    # 1. Initial progress message creation and sending
    #    cl.Message(content="") should have been called by invoke_langgraph
    mock_cl.Message.assert_any_call(content="")
    mock_progress_message_instance.send.assert_called_once()
    mock_progress_message_instance.stream_token.assert_any_call("⏳ Initializing InsightFlow process...")

    # 2. Progress message stored in user session
    mock_cl.user_session.set.assert_any_call("progress_msg", mock_progress_message_instance)

    # 3. InsightFlowCallbackHandler instantiation
    # Assert that the MagicMock representing the class constructor was called correctly.
    mock_actual_ifch_class_constructor.assert_called_once_with(progress_message=mock_progress_message_instance)
    assert passed_progress_msg_to_callback is mock_progress_message_instance # Ensure the correct msg object was captured

    # 4. Graph invocation with callback handler
    mock_graph_ainvoke.assert_called_once()
    # Check state passed to graph (first arg of ainvoke)
    actual_state_to_graph = mock_graph_ainvoke.call_args[0][0]
    assert actual_state_to_graph["query"] == mock_incoming_message.content
    # check config passed to graph (second arg of ainvoke, specifically callbacks)
    actual_config_to_graph = mock_graph_ainvoke.call_args.kwargs['config']
    assert len(actual_config_to_graph["callbacks"]) == 1
    assert isinstance(actual_config_to_graph["callbacks"][0], MockInsightFlowCallbackHandler)

    # 5. Final progress update
    mock_progress_message_instance.stream_token.assert_any_call("\n✨ InsightFlow processing complete!")
    mock_progress_message_instance.update.assert_called_once()
    
    # 6. Session `get` calls (original assertions)
    mock_cl.user_session.get.assert_any_call("direct_mode")
    mock_cl.user_session.get.assert_any_call("insight_flow_state") 

@pytest.mark.asyncio
async def test_invoke_langgraph_orchestrates_persona_execution(mock_cl):
    """
    Test that invoke_langgraph correctly processes a query through the graph,
    leading to execute_persona_tasks calling personas with their designated LLMs.
    Mocks are placed at the LLM's astream method.
    """
    query = "Tell me about black holes."
    initial_selected_personas = ["analytical", "scientific"]
    initial_insight_state = InsightFlowState(
        query="", # Query will be set by invoke_langgraph
        selected_personas=initial_selected_personas,
        persona_responses={},
        synthesized_response=None,
        visualization_code=None,
        visualization_image_url=None,
        current_step_name="awaiting_query",
        error_message=None,
        panel_type="research"
    )

    # --- Mock cl.user_session.get specifically for this test ---
    mock_persona_factory_instance = MagicMock(spec=PersonaFactory) # From utils.persona
    
    # Mock the create_persona method on the factory instance
    # It should now return *actual* PersonaReasoning instances, initialized with the passed (mocked) LLMs.
    def actual_create_persona_side_effect(persona_id, llm_instance):
        # This side effect will mimic the real factory's behavior of creating real PersonaReasoning objects,
        # but using the llm_instance provided (which will be one of our fully mocked LLMs).
        # We need the system prompts for the real PersonaReasoning constructor.
        # For simplicity in this test, we can use dummy prompts or fetch from a minimal config.
        # Or, ensure the mock_persona_factory_instance has a minimal persona_configs attribute for this.
        # Let's assume PersonaReasoning can be created with the llm_instance directly for this test purpose
        # if we mock its internal config loading or provide a simplified one.

        # The key is that it returns a REAL PersonaReasoning that will call .astream on the llm_instance it received.
        if persona_id == "analytical":
            return PersonaReasoning(persona_id="analytical", name="Analytical (Test)", system_prompt="Analytical System Prompt", llm=llm_instance)
        elif persona_id == "scientific":
            return PersonaReasoning(persona_id="scientific", name="Scientific (Test)", system_prompt="Scientific System Prompt", llm=llm_instance)
        return None
    
    # Ensure mock_persona_factory_instance is a spec of PersonaFactory so it has create_persona
    mock_persona_factory_instance.create_persona = MagicMock(side_effect=actual_create_persona_side_effect)

    session_data = {
        "persona_factory": mock_persona_factory_instance,
        "id": "test_thread_id",
        # "insight_flow_state": initial_insight_state # Not needed for invoke_langgraph direct call
    }
    mock_cl.user_session.get.side_effect = lambda key, default=None: session_data.get(key, default)
    
    # --- Mock cl.Message for progress and results messages --- #
    # Mock for the progress message created in invoke_langgraph
    mock_progress_msg_invoke_lg = AsyncMock(spec=cl.Message)
    mock_progress_msg_invoke_lg.send = AsyncMock()
    mock_progress_msg_invoke_lg.stream_token = AsyncMock()
    mock_progress_msg_invoke_lg.update = AsyncMock()

    # Mock for messages created in present_results (or other general messages)
    mock_other_msg_instance = AsyncMock(spec=cl.Message)
    mock_other_msg_instance.send = AsyncMock()
    # Add other methods like .stream_token or .update if present_results uses them directly
    # For now, send is the primary one asserted for present_results in this test.

    def message_constructor_side_effect(*args, **kwargs):
        if kwargs.get("content") == "": # Progress message from invoke_langgraph
            return mock_progress_msg_invoke_lg
        # Here, you might add more conditions if present_results creates messages
        # with specific content that needs a different mock. For now, this default works.
        return mock_other_msg_instance

    mock_cl.Message.side_effect = message_constructor_side_effect
    # Ensure mock_cl.Message class itself can be checked for calls like assert_any_call(content="")
    # We also need to reset call counts if mock_cl is shared across tests and cl.Message was called before.
    mock_cl.Message.reset_mock() 

    # --- Patch LLMs used by PersonaReasoning via PERSONA_LLM_MAP in app.py ---
    # We need to patch the actual LLM instances in app.py that PersonaReasoning will use.
    # The PersonaReasoning objects themselves are created *during* the graph run.
    # So, we mock the .astream method on the LLMs defined in app.py's PERSONA_LLM_MAP

    # Define mock stream behavior
    async def mock_llm_astream_analytical(*args, **kwargs):
        yield AIMessageChunk(content="Analytical perspective ")
        yield AIMessageChunk(content="on black holes.")

    async def mock_llm_astream_scientific(*args, **kwargs):
        yield AIMessageChunk(content="Scientific perspective ")
        yield AIMessageChunk(content="on black holes.")

    # --- Custom Async Iterator for mocking --- 
    class MockAsyncIterator:
        def __init__(self, items_or_generator_func, *args, **kwargs):
            # If a generator func is passed, call it to get the generator
            if inspect.isasyncgenfunction(items_or_generator_func):
                self.async_generator = items_or_generator_func(*args, **kwargs)
            elif inspect.isasyncgen(items_or_generator_func): # if already a generator object
                self.async_generator = items_or_generator_func
            else: # assume it's a list of items to be wrapped
                async def _gen():
                    for item in items_or_generator_func:
                        yield item
                self.async_generator = _gen()

        def __aiter__(self):
            return self.async_generator # The async_generator itself is the iterator

        async def __anext__(self):
            # This is not strictly needed if __aiter__ returns a proper async generator
            # as the generator handles its own __anext__.
            # However, to be a complete async iterator, it could be defined.
            # For now, relying on the returned async_generator from __aiter__.
            raise NotImplementedError # Should not be called if __aiter__ returns a true async gen

    # --- Function wrappers to ensure AsyncMock side_effect returns the async generator object directly ---
    def analytical_astream_wrapper(*args, **kwargs):
        return mock_llm_astream_analytical(*args, **kwargs)
    
    def scientific_astream_wrapper(*args, **kwargs):
        return mock_llm_astream_scientific(*args, **kwargs)

    # Patching the astream method of the LLMs by replacing them in app.PERSONA_LLM_MAP
    with patch.object(app, 'cl', new=mock_cl) as mock_cl_in_app:
        from app import invoke_langgraph, PERSONA_LLM_MAP, llm_analytical, llm_scientific, llm_synthesizer, initialize_configurations # Import map and original LLMs for type reference
        
        # Ensure configurations are initialized so llm_synthesizer is not None
        initialize_configurations() 

        original_persona_llm_map = PERSONA_LLM_MAP.copy()
        test_persona_llm_map = PERSONA_LLM_MAP.copy()

        mock_analytical_llm_replacement = MagicMock(spec=ChatOpenAI)
        analytical_astream_method_mock = AsyncMock(
            return_value=MockAsyncIterator(mock_llm_astream_analytical)
        )
        mock_analytical_llm_replacement.astream = analytical_astream_method_mock

        mock_scientific_llm_replacement = MagicMock(spec=ChatOpenAI)
        scientific_astream_method_mock = AsyncMock(
            return_value=MockAsyncIterator(mock_llm_astream_scientific)
        )
        mock_scientific_llm_replacement.astream = scientific_astream_method_mock

        test_persona_llm_map["analytical"] = mock_analytical_llm_replacement
        test_persona_llm_map["scientific"] = mock_scientific_llm_replacement

        # Mock for llm_synthesizer.ainvoke
        mock_synthesizer_response = MagicMock()
        mock_synthesizer_response.content = "Synthesized view of black holes."
        
        # Directly mock the ainvoke method on the app.llm_synthesizer instance
        # This needs app.llm_synthesizer to be already initialized.
        original_synthesizer_ainvoke = app.llm_synthesizer.ainvoke # Store original for restoration
        app.llm_synthesizer.ainvoke = AsyncMock(return_value=mock_synthesizer_response)

        # Patch the PERSONA_LLM_MAP in the app module.
        # The ChatOpenAI.ainvoke class patch is removed as we are direct mocking instance.
        mock_dalle_url = "http://fake_dalle_url.com/image.png"
        mock_mermaid_code = "graph TD; A-->B;"

        try:
            with patch.dict(sys.modules['app'].__dict__, {"PERSONA_LLM_MAP": test_persona_llm_map}):
                # Class patch for ChatOpenAI.ainvoke is removed here
                with patch(f'{APP_PY_PATH}.generate_dalle_image', AsyncMock(return_value=mock_dalle_url)) as mock_gen_dalle:
                    with patch(f'{APP_PY_PATH}.generate_mermaid_code', AsyncMock(return_value=mock_mermaid_code)) as mock_gen_mermaid:
                        final_state = await invoke_langgraph(query, initial_insight_state)
        finally:
            # Restore the original ainvoke method
            app.llm_synthesizer.ainvoke = original_synthesizer_ainvoke

    # Assertions
    mock_cl_in_app.user_session.get.assert_any_call("persona_factory")
    mock_cl_in_app.user_session.get.assert_any_call("id", "default_thread_id")

    # Check that create_persona on the factory was called correctly
    # This was part of the original test logic, but the main check is that the LLMs get called.
    # We can verify calls to the mock_persona_factory_instance.create_persona if needed,
    # ensuring it was called with the correct persona_id and the *actual* LLM instance from app.PERSONA_LLM_MAP
    # For now, primary assertion is on the LLM astream calls.
    mock_persona_factory_instance.create_persona.assert_any_call("analytical", mock_analytical_llm_replacement)
    mock_persona_factory_instance.create_persona.assert_any_call("scientific", mock_scientific_llm_replacement)

    # Verify the astream method on our *replacement mock LLM instances* was called.
    analytical_astream_method_mock.assert_called_once()
    scientific_astream_method_mock.assert_called_once()

    # Check final state content
    assert final_state["query"] == query
    assert "analytical" in final_state["persona_responses"]
    assert final_state["persona_responses"]["analytical"] == "Analytical perspective on black holes."
    assert "scientific" in final_state["persona_responses"]
    assert final_state["persona_responses"]["scientific"] == "Scientific perspective on black holes."
    assert final_state["current_step_name"] == "results_presented"
    assert "Synthesized view" in final_state.get("synthesized_response", "") # Check it contains the key part
    assert final_state.get("visualization_code") == mock_mermaid_code # Verify mocked mermaid code
    assert final_state.get("visualization_image_url") == mock_dalle_url # Verify mocked DALL-E URL
    
    # --- Assertions for Progress Message from invoke_langgraph ---
    mock_cl.Message.assert_any_call(content="") # Initial progress message creation
    mock_progress_msg_invoke_lg.send.assert_called_once() # Sent initially
    mock_progress_msg_invoke_lg.stream_token.assert_any_call("⏳ Initializing InsightFlow process...")
    # Add assertion for session set if invoke_langgraph sets progress_msg in session for this test's path
    # mock_cl.user_session.set.assert_any_call("progress_msg", mock_progress_msg_invoke_lg)
    mock_progress_msg_invoke_lg.stream_token.assert_any_call("\n✨ InsightFlow processing complete!")
    mock_progress_msg_invoke_lg.update.assert_called_once()

    # Check that present_results sent a message (using the mock_other_msg_instance)
    # This assertion assumes present_results creates one message. If it creates multiple, this needs adjustment.
    # If present_results creates messages with different content, the side_effect might need to be more specific.
    assert mock_other_msg_instance.send.called # Check if send was called on the message from present_results
    # If only one message is expected from present_results:
    # mock_other_msg_instance.send.assert_called_once() 

    # --- Assertions for Visualization Function Calls ---
    mock_gen_dalle.assert_called_once()
    # We can add more specific assertions about the arguments if needed, e.g.,
    # mock_gen_dalle.assert_called_once_with(prompt=ANY, client=ANY)
    mock_gen_mermaid.assert_called_once()
    # mock_gen_mermaid.assert_called_once_with(text_input=ANY, llm_client=ANY)

@pytest.mark.asyncio
async def test_on_message_quick_mode_on_overrides_personas(mock_cl):
    """Test that on_message with quick_mode=True overrides selected_personas before calling invoke_langgraph."""
    mock_incoming_message = MagicMock(spec=cl.Message)
    mock_incoming_message.content = "Test query for quick mode on"

    initial_personas = ["creative", "historical"]
    initial_state_dict = {
        "query": "", "selected_personas": initial_personas, "persona_responses": {},
        "synthesized_response": None, "visualization_code": None,
        "visualization_image_url": None, "current_step_name": "awaiting_query",
        "error_message": None, "panel_type": "research"
    }
    
    setup_mock_cl_session_get(mock_cl, direct_mode_value=False, quick_mode_value=True, initial_state_dict=initial_state_dict)
    
    # --- Mock cl.Message for progress updates (similar to test_on_message_direct_mode_off_calls_langgraph) --- #
    mock_progress_message_instance = AsyncMock(spec=cl.Message)
    mock_progress_message_instance.send = AsyncMock()
    mock_progress_message_instance.stream_token = AsyncMock()
    mock_progress_message_instance.update = AsyncMock()

    default_mock_other_message_instance = AsyncMock(spec=cl.Message, send=AsyncMock())
    def message_side_effect_quick_on(*args, **kwargs):
        if kwargs.get("content") == "": 
            return mock_progress_message_instance
        return default_mock_other_message_instance
    mock_cl.Message.side_effect = message_side_effect_quick_on
    mock_cl.Message.reset_mock() # Reset from other tests

    expected_quick_mode_personas = ["test_quick1", "test_quick2"]
    passed_progress_msg_to_callback_quick_on = None

    # Define a unique local mock handler class for this test to avoid nonlocal conflicts
    class MockCBHandlerQuickOnLocal(app.InsightFlowCallbackHandler): # Changed name
        def __init__(self, progress_message: cl.Message):
            nonlocal passed_progress_msg_to_callback_quick_on
            passed_progress_msg_to_callback_quick_on = progress_message
            # super().__init__(progress_message) # Not strictly necessary for the mock's role
            self.progress_message = progress_message

    def mock_ifch_constructor_side_effect_quick_on(progress_message):
        return MockCBHandlerQuickOnLocal(progress_message=progress_message)

    with patch(f'{APP_PY_PATH}.cl', new=mock_cl):
        with patch(f'{APP_PY_PATH}.QUICK_MODE_PERSONAS', new=expected_quick_mode_personas):
            with patch(f'{APP_PY_PATH}.insight_flow_graph.ainvoke', new_callable=AsyncMock) as mock_graph_ainvoke:
                # Patch InsightFlowCallbackHandler with a MagicMock whose side_effect instantiates the local mock
                with patch(f'{APP_PY_PATH}.InsightFlowCallbackHandler') as mock_ifch_constructor_quick_on:
                    mock_ifch_constructor_quick_on.side_effect = mock_ifch_constructor_side_effect_quick_on
                    from app import on_message
                    await on_message(mock_incoming_message)

    # Progress Message Assertions
    mock_cl.Message.assert_any_call(content="")
    mock_progress_message_instance.send.assert_called_once()
    mock_progress_message_instance.stream_token.assert_any_call("⏳ Initializing InsightFlow process...")
    mock_cl.user_session.set.assert_any_call("progress_msg", mock_progress_message_instance)
    # Assert on the MagicMock that replaced the class constructor
    mock_ifch_constructor_quick_on.assert_called_once_with(progress_message=mock_progress_message_instance)
    assert passed_progress_msg_to_callback_quick_on is mock_progress_message_instance # Check captured arg
    mock_progress_message_instance.stream_token.assert_any_call("\n✨ InsightFlow processing complete!")
    mock_progress_message_instance.update.assert_called_once()

    # Original Assertions
    mock_graph_ainvoke.assert_called_once()
    called_state = mock_graph_ainvoke.call_args[0][0]
    assert called_state["selected_personas"] == expected_quick_mode_personas
    assert called_state["query"] == mock_incoming_message.content
    mock_cl.user_session.get.assert_any_call("direct_mode")
    mock_cl.user_session.get.assert_any_call("quick_mode", False)
    mock_cl.user_session.get.assert_any_call("insight_flow_state")

@pytest.mark.asyncio
async def test_on_message_quick_mode_off_uses_original_personas(mock_cl):
    """Test that on_message with quick_mode=False uses original selected_personas for invoke_langgraph."""
    mock_incoming_message = MagicMock(spec=cl.Message)
    mock_incoming_message.content = "Test query for quick mode off"

    original_selected_personas = ["original1", "original2"]
    initial_state_dict_off = {
        "query": "", "selected_personas": original_selected_personas, "persona_responses": {},
        "synthesized_response": None, "visualization_code": None,
        "visualization_image_url": None, "current_step_name": "awaiting_query",
        "error_message": None, "panel_type": "research"
    }
    
    setup_mock_cl_session_get(mock_cl, direct_mode_value=False, quick_mode_value=False, initial_state_dict=initial_state_dict_off)
    
    mock_progress_message_instance_off = AsyncMock(spec=cl.Message)
    mock_progress_message_instance_off.send = AsyncMock()
    mock_progress_message_instance_off.stream_token = AsyncMock()
    mock_progress_message_instance_off.update = AsyncMock()

    default_mock_other_message_instance_off = AsyncMock(spec=cl.Message, send=AsyncMock())
    def message_side_effect_quick_off(*args, **kwargs):
        if kwargs.get("content") == "": 
            return mock_progress_message_instance_off
        return default_mock_other_message_instance_off
    mock_cl.Message.side_effect = message_side_effect_quick_off
    mock_cl.Message.reset_mock()

    passed_progress_msg_to_callback_quick_off = None
    # Define a unique local mock handler class for this test
    class MockCBHandlerQuickOffLocal(app.InsightFlowCallbackHandler): # Changed name
        def __init__(self, progress_message: cl.Message):
            nonlocal passed_progress_msg_to_callback_quick_off
            passed_progress_msg_to_callback_quick_off = progress_message
            # super().__init__(progress_message) # Not strictly necessary
            self.progress_message = progress_message

    def mock_ifch_constructor_side_effect_quick_off(progress_message):
        return MockCBHandlerQuickOffLocal(progress_message=progress_message)

    with patch(f'{APP_PY_PATH}.cl', new=mock_cl):
        with patch(f'{APP_PY_PATH}.insight_flow_graph.ainvoke', new_callable=AsyncMock) as mock_graph_ainvoke:
            # Patch InsightFlowCallbackHandler with a MagicMock whose side_effect instantiates the local mock
            with patch(f'{APP_PY_PATH}.InsightFlowCallbackHandler') as mock_ifch_constructor_quick_off:
                mock_ifch_constructor_quick_off.side_effect = mock_ifch_constructor_side_effect_quick_off
                from app import on_message
                await on_message(mock_incoming_message)

    # Progress Message Assertions
    mock_cl.Message.assert_any_call(content="")
    mock_progress_message_instance_off.send.assert_called_once()
    mock_progress_message_instance_off.stream_token.assert_any_call("⏳ Initializing InsightFlow process...")
    mock_cl.user_session.set.assert_any_call("progress_msg", mock_progress_message_instance_off)
    # Assert on the MagicMock that replaced the class constructor
    mock_ifch_constructor_quick_off.assert_called_once_with(progress_message=mock_progress_message_instance_off)
    assert passed_progress_msg_to_callback_quick_off is mock_progress_message_instance_off # Check captured arg
    mock_progress_message_instance_off.stream_token.assert_any_call("\n✨ InsightFlow processing complete!")
    mock_progress_message_instance_off.update.assert_called_once()

    # Original Assertions
    mock_graph_ainvoke.assert_called_once()
    called_state = mock_graph_ainvoke.call_args[0][0]
    assert called_state["selected_personas"] == original_selected_personas
    assert called_state["query"] == mock_incoming_message.content
    mock_cl.user_session.get.assert_any_call("direct_mode")
    mock_cl.user_session.get.assert_any_call("quick_mode", False)
    mock_cl.user_session.get.assert_any_call("insight_flow_state") 