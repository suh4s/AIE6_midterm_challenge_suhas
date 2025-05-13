# Utility functions for InsightFlow AI

# Placeholder for state type, assuming it's a dict-like object or a dataclass
# from state import InsightFlowState # Or from app import InsightFlowState if defined there

def toggle_direct_mode(state: dict):
    """Synchronously toggles the 'direct_mode' boolean in the state dictionary."""
    state['direct_mode'] = not state.get('direct_mode', False)
    # This function purely mutates the state. 
    # UI updates (sending messages, updating settings panel) should be handled by the caller (e.g., Chainlit action callback)

# Add other utility functions here as needed, for example:
# - Persona management utilities
# - Helper for formatting messages
# - etc. 