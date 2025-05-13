"""
Implementations of different persona reasoning types.
"""

from .base import PersonaReasoning
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing import Dict, List, Optional, Any
from langchain_core.documents import Document

class LLMPersonaReasoning(PersonaReasoning):
    """Base implementation that uses LLM to generate responses"""
    
    def __init__(self, config: Dict[str, Any], llm=None):
        super().__init__(config)
        # Use shared LLM instance if provided, otherwise create one
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
        
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        """Generate perspective using LLM with persona's system prompt"""
        
        # Build prompt with context if available
        context_text = ""
        if context and len(context) > 0:
            context_text = "\n\nRelevant information:\n" + "\n".join([doc.page_content for doc in context])
            
        # Build messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Query: {query}{context_text}\n\nPlease provide your perspective on this query based on your unique approach.")
        ]
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        return response.content

# Specialized implementations for each persona type
class AnalyticalReasoning(LLMPersonaReasoning):
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        """Generate perspective using analytical reasoning approach"""
        # For MVP, we'll use the base implementation
        # In a full implementation, add analytical-specific enhancements
        return super().generate_perspective(query, context)

class ScientificReasoning(LLMPersonaReasoning):
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        """Generate perspective using scientific reasoning approach"""
        # For MVP, we'll use the base implementation
        # In a full implementation, add scientific-specific enhancements
        return super().generate_perspective(query, context)

class PhilosophicalReasoning(LLMPersonaReasoning):
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        """Generate perspective using philosophical reasoning approach"""
        # For MVP, we'll use the base implementation
        # In a full implementation, add philosophical-specific enhancements
        return super().generate_perspective(query, context)

class FactualReasoning(LLMPersonaReasoning):
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        """Generate perspective using factual reasoning approach"""
        # For MVP, we'll use the base implementation
        # In a full implementation, add factual-specific enhancements
        return super().generate_perspective(query, context)

class MetaphoricalReasoning(LLMPersonaReasoning):
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        """Generate perspective using metaphorical reasoning approach"""
        # For MVP, we'll use the base implementation
        # In a full implementation, add metaphorical-specific enhancements
        return super().generate_perspective(query, context)

class FuturisticReasoning(LLMPersonaReasoning):
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        """Generate perspective using futuristic reasoning approach"""
        # For MVP, we'll use the base implementation
        # In a full implementation, add futuristic-specific enhancements
        return super().generate_perspective(query, context)

# Personality implementations (second tier of two-tier system)
class HolmesReasoning(LLMPersonaReasoning):
    """Sherlock Holmes personality implementation"""
    
    def __init__(self, config: Dict[str, Any], parent_config: Dict[str, Any], llm=None):
        super().__init__(config, llm)
        self.parent_config = parent_config
        
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        """Generate perspective in Sherlock Holmes' style"""
        # For MVP, we'll use the base implementation with Holmes' system prompt
        # In a full implementation, add Holmes-specific reasoning patterns
        return super().generate_perspective(query, context)

class FeynmanReasoning(LLMPersonaReasoning):
    """Richard Feynman personality implementation"""
    
    def __init__(self, config: Dict[str, Any], parent_config: Dict[str, Any], llm=None):
        super().__init__(config, llm)
        self.parent_config = parent_config
        
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        """Generate perspective in Richard Feynman's style"""
        # For MVP, we'll use the base implementation with Feynman's system prompt
        # In a full implementation, add Feynman-specific reasoning patterns
        return super().generate_perspective(query, context)

class FryReasoning(LLMPersonaReasoning):
    """Hannah Fry personality implementation"""
    
    def __init__(self, config: Dict[str, Any], parent_config: Dict[str, Any], llm=None):
        super().__init__(config, llm)
        self.parent_config = parent_config
        
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        """Generate perspective in Hannah Fry's style"""
        # For MVP, we'll use the base implementation with Fry's system prompt
        # In a full implementation, add Fry-specific reasoning patterns
        return super().generate_perspective(query, context) 