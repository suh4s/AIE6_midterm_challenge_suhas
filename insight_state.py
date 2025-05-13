"""
State management for InsightFlow AI.
"""

from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.documents import Document

class InsightFlowState(TypedDict):
    """
    State for InsightFlow AI.
    
    This state is used by LangGraph to track the current state of the system.
    """
    # Query information
    panel_type: str  # "research" or "discussion"
    query: str
    selected_personas: List[str]
    
    # Research results
    persona_responses: Dict[str, str]
    synthesized_response: Optional[str]
    visualization_code: Optional[str]  # For storing Mermaid diagram code
    visualization_image_url: Optional[str]  # For storing DALL-E generated image URL
    
    # Control
    current_step_name: str
    error_message: Optional[str] 