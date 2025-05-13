import pytest
from unittest.mock import patch, MagicMock
import importlib
from collections import Counter

# Patch load_dotenv where it's looked up by app.py (its source module)
@patch('dotenv.load_dotenv') 
@patch('langchain_openai.ChatOpenAI') # Patch ChatOpenAI at its source module
@patch('openai.AsyncOpenAI')          # Patch AsyncOpenAI at its source module
def test_llm_initialization(mock_async_openai_class, mock_chat_openai_class, mock_load_dotenv_func):
    """Test that required LLM instances and OpenAI client are initialized correctly."""
    # mock_load_dotenv_func is the mock for dotenv.load_dotenv
    # mock_chat_openai_class is the mock for langchain_openai.ChatOpenAI
    # mock_async_openai_class is the mock for openai.AsyncOpenAI
    
    import app 
    importlib.reload(app) # Force reload app module AFTER mocks are applied
    app.initialize_configurations() # Call the new initialization function

    # Check that load_dotenv was called
    assert mock_load_dotenv_func.called

    # Check that AsyncOpenAI client was initialized
    assert mock_async_openai_class.called
    mock_async_openai_class.assert_called_once()

    expected_configs = {
        'llm_planner': {'model': 'gpt-3.5-turbo', 'temperature': 0.1},
        'llm_synthesizer': {'model': 'gpt-4o-mini', 'temperature': 0.4},
        'llm_direct': {'model': 'gpt-3.5-turbo', 'temperature': 0.3},
        'llm_analytical': {'model': 'gpt-3.5-turbo', 'temperature': 0.2},
        'llm_scientific': {'model': 'gpt-3.5-turbo', 'temperature': 0.3},
        'llm_philosophical': {'model': 'gpt-4o-mini', 'temperature': 0.5},
        'llm_factual': {'model': 'gpt-3.5-turbo', 'temperature': 0.1},
        'llm_metaphorical': {'model': 'gpt-4o-mini', 'temperature': 0.6},
        'llm_futuristic': {'model': 'gpt-4o-mini', 'temperature': 0.6},
        'llm_mermaid_generator': {'model': 'gpt-3.5-turbo', 'temperature': 0.1},
    }

    call_args_list = mock_chat_openai_class.call_args_list

    # Expect each of the 10 configs to be called once
    expected_total_calls = len(expected_configs)
    assert len(call_args_list) == expected_total_calls, \
        f"Expected {expected_total_calls} LLM initializations, but found {len(call_args_list)}"

    actual_calls_configs = []
    for call in call_args_list:
        _, kwargs = call
        actual_calls_configs.append({'model': kwargs.get('model'), 'temperature': kwargs.get('temperature')})

    # Convert dicts to a hashable type (tuple of sorted items) for Counter
    actual_configs_tuples = [tuple(sorted(c.items())) for c in actual_calls_configs]
    actual_counts = Counter(actual_configs_tuples)

    # Count how many times each unique configuration is expected based on the definitions
    # This accounts for the same config being used for multiple LLM variables (e.g. planner and factual)
    expected_config_definitions_as_tuples = [tuple(sorted(c.items())) for c in expected_configs.values()]
    expected_definition_counts = Counter(expected_config_definitions_as_tuples)

    for config_tuple, num_definitions in expected_definition_counts.items():
        # Each definition now runs only once
        expected_actual_calls_for_this_config = num_definitions 
        assert actual_counts[config_tuple] == expected_actual_calls_for_this_config, \
            f"Expected config {dict(config_tuple)} (defined {num_definitions} times) to be actually called {expected_actual_calls_for_this_config} times, but found {actual_counts[config_tuple]} times."

    assert len(actual_counts) == len(expected_definition_counts), \
        f"Found {len(actual_counts)} unique LLM configurations in actual calls, but expected {len(expected_definition_counts)} unique configurations to be defined." 