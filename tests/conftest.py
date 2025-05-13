import pytest
from unittest.mock import MagicMock, patch
from unittest.mock import AsyncMock

@pytest.fixture
def mock_chat_openai():
    """Fixture to mock langchain_openai.ChatOpenAI."""
    mock = MagicMock()
    # Simulate the behavior of the ChatOpenAI constructor if needed
    mock.return_value = MagicMock() 
    return mock

@pytest.fixture
def mock_cl():
    """Fixture to mock the chainlit library."""
    mock = MagicMock() # Removed spec=True
    mock.__name__ = "chainlit" # Act more like a module
    # mock.__file__ = "mocked_chainlit.py" # Another common module attribute

    # Mock specific chainlit attributes or methods if tests rely on them
    mock.user_session = MagicMock()
    mock.user_session.set = MagicMock()
    mock.Message = MagicMock()
    # Ensure send() is an AsyncMock if it needs to be awaited and its result checked
    mock.Message.return_value.send = AsyncMock() 

    # Mock cl.Action as it will be instantiated by the app
    mock.Action = MagicMock() # Allows app to call cl.Action(...)

    # Mock cl.Select as it will be instantiated by the app for persona choices
    mock.Select = MagicMock() # Allows app to call cl.Select(...)

    # Mock cl.Switch for UI toggles
    mock.Switch = MagicMock()

    # Mock cl.ChatSettings for UI settings panel
    mock.ChatSettings = MagicMock() # Mock the ChatSettings class
    # Ensure the instance returned by ChatSettings() has an async send method
    mock.ChatSettings.return_value.send = AsyncMock()

    # Mock decorators like on_chat_start and on_message
    # They need to be callable and return the function they decorate.
    mock.on_chat_start = MagicMock(side_effect=lambda func: func)
    mock.on_message = MagicMock(side_effect=lambda func: func)
    
    return mock

@pytest.fixture
def mock_dotenv_load():
    """Fixture to mock dotenv.load_dotenv."""
    mock = MagicMock()
    return mock

# It's good practice to also provide the mock StateGraph classes here
# if they are used across multiple test files for langgraph setup.

@pytest.fixture
def mock_state_graph_instance():
    """Returns a mock instance of StateGraph, typically the one returned by the class mock."""
    instance = MagicMock(spec=['add_node', 'set_entry_point', 'add_conditional_edges', 'add_edge', 'compile'])
    # If add_node returns self or another mock, configure here if needed for chaining
    return instance

@pytest.fixture
def mock_state_graph_class(mock_state_graph_instance):
    """Fixture to mock the StateGraph class itself."""
    mock = MagicMock(return_value=mock_state_graph_instance)
    return mock 

@pytest.fixture
def mock_chainlit_context(mock_cl): # Depends on mock_cl to use its user_session mock
    """Mocks chainlit.context.context_var.get() to return a mock context with a session."""
    mock_context_obj = MagicMock(name="MockChainlitContextObject")
    # Use the same user_session mock from the mock_cl fixture for consistency
    mock_context_obj.session = mock_cl.user_session 

    with patch('chainlit.context.context_var.get', return_value=mock_context_obj) as mock_context_var_get:
        yield mock_context_var_get 