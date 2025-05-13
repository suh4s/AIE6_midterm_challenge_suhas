import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app import synthesize_responses # Removed generate_visualization from here as it's the SUT for one test
from insight_state import InsightFlowState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI # For spec
from openai import AsyncOpenAI # For spec of the client if we were to patch app.openai_async_client
# Import app to allow patching its global openai_async_client
import app as application_module 

# Define a plausible system prompt for the synthesizer for use in assertion
SYNTHESIZER_SYSTEM_PROMPT_TEMPLATE = """You are a master synthesizer AI. Your task is to integrate the following diverse perspectives into a single, coherent, and insightful response. Ensure that the final synthesis is well-structured, easy to understand, and accurately reflects the nuances of each provided viewpoint. Do not simply list the perspectives; weave them together.

Perspectives:
{formatted_perspectives}

Synthesized Response:"""

@pytest.mark.asyncio
async def test_synthesize_responses_node():
    """
    Test the synthesize_responses node to ensure it calls the llm_synthesizer
    with the correct prompt and updates the state.
    """
    initial_state = InsightFlowState(
        query="Test Query",
        selected_personas=["analytical", "scientific"],
        persona_responses={
            "analytical": "Analytical perspective text.",
            "scientific": "Scientific perspective text."
        },
        synthesized_response=None,
        current_step_name="execute_persona_tasks" # Previous step
    )

    mock_synthesized_text = "This is the beautifully synthesized response from the LLM."
    
    # Expected formatted perspectives string
    expected_perspectives_text = ("- Perspective from analytical: Analytical perspective text.\n"
                                  "- Perspective from scientific: Scientific perspective text.")
    expected_final_prompt_content = SYNTHESIZER_SYSTEM_PROMPT_TEMPLATE.format(formatted_perspectives=expected_perspectives_text)

    # Create a mock for the entire llm_synthesizer object in app.py
    mock_llm_synthesizer_replacement = MagicMock(spec=ChatOpenAI)
    # Mock its ainvoke method
    mock_llm_synthesizer_replacement.ainvoke = AsyncMock(return_value=AIMessage(content=mock_synthesized_text))

    # Patch app.llm_synthesizer to be our mock_llm_synthesizer_replacement
    with patch('app.llm_synthesizer', new=mock_llm_synthesizer_replacement) as mock_synthesizer_in_app:
        # Call the actual node function
        # synthesize_responses (from app) will now use the patched llm_synthesizer (mock_synthesizer_in_app)
        output_state = await synthesize_responses(initial_state.copy())

    # Assertions
    mock_synthesizer_in_app.ainvoke.assert_called_once()
    args, kwargs = mock_synthesizer_in_app.ainvoke.call_args
    called_messages = args[0] # The 'messages' argument
    
    assert len(called_messages) == 1 # Expecting a single system message for simplicity here, or system + human
    # For now, let's assume the prompt is constructed as a single SystemMessage containing everything
    # This might evolve when we implement the actual prompt construction in app.py
    assert isinstance(called_messages[0], SystemMessage) 
    assert called_messages[0].content == expected_final_prompt_content
    
    assert output_state["synthesized_response"] == mock_synthesized_text
    assert output_state["current_step_name"] == "generate_visualization"
    assert output_state["persona_responses"] == initial_state["persona_responses"] # Should not change 

@pytest.mark.asyncio
@patch('app.generate_dalle_image') # Corrected: Patch where it's used by app.generate_visualization
@patch('app.openai_async_client') 
@patch('app.generate_mermaid_code') # <--- ADDED PATCH for mermaid generation utility
async def test_generate_visualization_node_creates_dalle_and_mermaid(
    mock_app_generate_mermaid_code, # <--- ADDED mock argument
    mock_app_openai_client, 
    mock_app_level_generate_dalle_image_func # Renamed to reflect it's app's view of the function
):
    """
    Test the generate_visualization node to ensure it calls generate_dalle_image
    and generate_mermaid_code, and updates the state with image URL and mermaid code.
    """
    initial_synthesized_response = "This is a detailed synthesized response about complex topics."
    initial_state = InsightFlowState(
        query="Test Query for Visuals",
        synthesized_response=initial_synthesized_response,
        persona_responses={}, # Explicitly initialize for completeness
        visualization_image_url=None,
        visualization_code=None,
        current_step_name="synthesize_responses" 
    )

    expected_dalle_url = "https://dalle.example.com/generated_image.png"
    mock_app_level_generate_dalle_image_func.return_value = expected_dalle_url
    
    expected_mermaid_output = "graph LR; A-->B; C-->D;" # Mocked mermaid output
    mock_app_generate_mermaid_code.return_value = expected_mermaid_output
    
    # Ensure llm_mermaid_generator is available in the application_module for the test context
    # If app.llm_mermaid_generator is None during the test, the call might be skipped.
    # We assume it's initialized by initialize_configurations(), which should run before/during app import or be called by on_chat_start.
    # For node tests, if direct LLM calls are made, ensure they are properly mocked or the LLMs are available.
    # Here, generate_mermaid_code (the util) takes the LLM as an arg, which app.generate_visualization supplies.
    # So, we need to make sure app.llm_mermaid_generator is a mock or a real object for the call.
    # Let's patch app.llm_mermaid_generator to be a MagicMock for this test to ensure it's passed correctly.
    with patch.object(application_module, 'llm_mermaid_generator', new_callable=MagicMock) as mock_llm_mermaid_in_app:
        output_state = await application_module.generate_visualization(initial_state.copy())

    # DALL-E Assertions
    mock_app_level_generate_dalle_image_func.assert_called_once()
    called_dalle_kwargs = mock_app_level_generate_dalle_image_func.call_args.kwargs
    assert called_dalle_kwargs.get('prompt') == f"A hand-drawn style visual note or sketch representing the key concepts of: {initial_synthesized_response}"
    assert called_dalle_kwargs.get('client') is mock_app_openai_client
    assert output_state["visualization_image_url"] == expected_dalle_url

    # Mermaid Assertions
    mock_app_generate_mermaid_code.assert_called_once_with(
        initial_synthesized_response, 
        mock_llm_mermaid_in_app # Assert it was called with the (mocked) app.llm_mermaid_generator
    )
    assert output_state["visualization_code"] == expected_mermaid_output 
    
    # General State Assertions
    assert output_state["current_step_name"] == "present_results"
    assert output_state["synthesized_response"] == initial_synthesized_response
    assert output_state["persona_responses"] == initial_state["persona_responses"] # Should not change 

@pytest.mark.asyncio
@patch('app.cl.user_session.get')
@patch('app.cl.Image') # Mock for cl.Image class
@patch('app.cl.Text')  # <--- ADDED PATCH for cl.Text class
@patch('app.cl.Message') # Mock for cl.Message class
async def test_present_results_node_sends_all_content(
    mock_message_class, 
    mock_text_class,    # <--- ADDED mock argument
    mock_image_class, 
    mock_user_session_get,
):
    """Test present_results node sends synthesized response, visuals, and perspectives."""
    
    # Configure app.cl.user_session.get mock
    def user_session_get_side_effect(key, default=None):
        if key == "show_visualization":
            return True
        if key == "show_perspectives":
            return True
        # Fallback for other keys used by the node if any (e.g. in future)
        # Currently, present_results from app.py doesn't use default values from cl.user_session.get for these booleans directly in its logic
        # It implies they should exist. The main app's on_chat_start sets them.
        # For this test, we only care about show_visualization and show_perspectives.
        return default
    mock_user_session_get.side_effect = user_session_get_side_effect

    # Configure cl.Message mock (the class itself)
    # Each time cl.Message() is called, it returns a new mock_message_instance.
    # This instance needs an async 'send' method.
    # To allow multiple calls to cl.Message and check them, we need Message instances to be distinct if their send methods are checked individually.
    # A simpler way for now is to check the call_args_list of mock_message_class.
    # The send method is on the instance, so we make the class return an instance that has an async send.
    async def mock_send(*args, **kwargs):
        pass # Mock async send
    
    # Store all created message instances to check their send calls individually if needed
    created_message_mocks = [] 
    def message_constructor_side_effect(*args, **kwargs):
        instance = MagicMock(name=f"MockClMessageInstance_{len(created_message_mocks)}")
        instance.send = AsyncMock(wraps=mock_send) # each instance gets its own send mock
        instance.content = kwargs.get("content")
        instance.elements = kwargs.get("elements")
        created_message_mocks.append(instance)
        return instance

    mock_message_class.side_effect = message_constructor_side_effect

    # Configure cl.Image mock (the class itself)
    mock_image_instance_returned_by_constructor = MagicMock(name="MockClImageInstance")
    mock_image_class.return_value = mock_image_instance_returned_by_constructor

    # Configure cl.Text mock (the class itself)
    mock_text_instance_returned_by_constructor = MagicMock(name="MockClTextInstance")
    mock_text_class.return_value = mock_text_instance_returned_by_constructor

    initial_state = application_module.InsightFlowState(
        query="Test Query for Presentation",
        synthesized_response="Final synthesized answer.",
        persona_responses={"analytical": "Analytical point.", "creative": "Creative idea."},
        visualization_image_url="http://example.com/dalle.png",
        visualization_code="graph TD; X-->Y;",
        current_step_name="generate_visualization" # Previous step
    )

    output_state = await application_module.present_results(initial_state.copy())

    # Assertions for user_session.get calls
    mock_user_session_get.assert_any_call("show_visualization")
    mock_user_session_get.assert_any_call("show_perspectives")

    # Assertions for cl.Message calls
    # Expected calls: 1 for synthesized, 1 for DALL-E, 1 for Mermaid, 2 for perspectives = 5 calls
    assert mock_message_class.call_count == 5
    # Ensure each created message mock had its send method called once
    for msg_mock_instance in created_message_mocks:
        msg_mock_instance.send.assert_called_once()

    all_message_constructor_calls = mock_message_class.call_args_list

    # 1. Synthesized response
    call_synthesized_kwargs = all_message_constructor_calls[0].kwargs
    assert call_synthesized_kwargs.get('content') == "Final synthesized answer."
    assert not call_synthesized_kwargs.get('elements') 

    # 2. DALL-E Image
    call_dalle_kwargs = all_message_constructor_calls[1].kwargs
    mock_image_class.assert_called_once_with(
        url="http://example.com/dalle.png", 
        name="dalle_visualization", 
        display="inline",
        size="large" # Default in app.py
    )
    assert call_dalle_kwargs.get('elements') == [mock_image_instance_returned_by_constructor]
    # Content for image message could be None or specific text. Let's assume it's empty if not specified.
    assert call_dalle_kwargs.get('content') == "" # Or check for a specific title if app.py adds one

    # 3. Mermaid Diagram
    call_mermaid_kwargs = all_message_constructor_calls[2].kwargs
    mock_text_class.assert_called_once_with(
        content="graph TD; X-->Y;", # The raw mermaid code from initial_state
        mime_type="text/mermaid",
        name="generated_diagram",
        display="inline"
    )
    assert call_mermaid_kwargs.get('elements') == [mock_text_instance_returned_by_constructor]
    assert call_mermaid_kwargs.get('content') == "" # Expecting empty content for message with element

    # 4. Persona Perspectives (order within perspectives might not be guaranteed due to dict iteration)
    perspective_contents_sent = set()
    perspective_contents_sent.add(all_message_constructor_calls[3].kwargs.get('content'))
    perspective_contents_sent.add(all_message_constructor_calls[4].kwargs.get('content'))

    expected_perspective1_content = "**Perspective from analytical:**\nAnalytical point."
    expected_perspective2_content = "**Perspective from creative:**\nCreative idea."
    assert expected_perspective1_content in perspective_contents_sent
    assert expected_perspective2_content in perspective_contents_sent
    
    # Check state update
    assert output_state["current_step_name"] == "results_presented"
