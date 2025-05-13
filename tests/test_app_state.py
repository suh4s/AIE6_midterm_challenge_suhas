import pytest
from unittest.mock import MagicMock

# Assuming InsightFlowState is importable and structured appropriately
# from state import InsightFlowState # If it were a dataclass
from utils.state_utils import toggle_direct_mode # Import from new utils/state_utils.py

# Mocking InsightFlowState for testing isolation
class MockInsightFlowState(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set initial default state relevant to the test
        self.setdefault('direct_mode', False)
        self.setdefault('selected_personas', ['analytical', 'scientific']) # Example default

# --- Test Case (no change to the test itself) ---
def test_toggle_direct_mode():
    """Tests that calling toggle_direct_mode flips the direct_mode state."""
    # Arrange: Create an initial state
    initial_state = MockInsightFlowState()
    assert initial_state['direct_mode'] is False # Verify initial state

    # Act: Simulate the toggle action (calling the function)
    toggle_direct_mode(initial_state)

    # Assert: Check if the state changed
    assert initial_state['direct_mode'] is True

    # Act: Toggle again
    toggle_direct_mode(initial_state)

    # Assert: Check if it toggled back
    assert initial_state['direct_mode'] is False 