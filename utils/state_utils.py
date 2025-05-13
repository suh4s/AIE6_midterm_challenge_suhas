# Utility functions for InsightFlow AI state management

from typing import Dict, Any

def toggle_direct_mode(state: Dict[str, Any]) -> None:
    """Toggles the 'direct_mode' boolean in the given state dictionary."""
    if 'direct_mode' not in state:
        state['direct_mode'] = True  # Initialize if not present, defaulting to True after first toggle
    else:
        state['direct_mode'] = not state['direct_mode'] 