"""
Base classes for the persona system.
"""

from typing import Dict, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel # For type hinting the LLM
from langchain_core.messages import SystemMessage, HumanMessage
import inspect

class PersonaReasoning:
    """Represents the reasoning capabilities of a persona."""
    def __init__(self, persona_id: str, name: str, system_prompt: str, llm: BaseChatModel):
        self.persona_id = persona_id
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm # Store the actual LLM instance
        print(f"DEBUG: PersonaReasoning for {self.name} initialized with LLM: {type(self.llm)}")

    async def generate_perspective(self, query: str) -> str:
        print(f"DEBUG: Generating perspective for {self.name} on query: '{query[:50]}...'")
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query)
        ]
        
        response_content = ""
        # Stream the response from the LLM
        # If self.llm.astream(messages) is an AsyncMock, it might return a coroutine that yields the iterator.
        stream_source = self.llm.astream(messages)
        if inspect.iscoroutine(stream_source):
            async_iterator = await stream_source
        else:
            async_iterator = stream_source
            
        async for chunk in async_iterator:
            if chunk.content:
                response_content += chunk.content
        
        print(f"DEBUG: Perspective from {self.name}: '{response_content[:100]}...'")
        return response_content

class PersonaFactory:
    """Factory for creating persona instances."""
    def __init__(self, config_dir: str = "persona_configs"):
        self.config_dir = config_dir
        # Configs now store parameters, not direct LLM configs as LLMs are passed in
        self.persona_configs: Dict[str, Dict[str, Any]] = self._load_persona_configs()
        print(f"DEBUG: PersonaFactory initialized. Loaded {len(self.persona_configs)} persona base configs from {config_dir}.")

    def _load_persona_configs(self) -> Dict[str, Dict[str, Any]]:
        # These configs now only store name and system_prompt.
        # The LLM to be used is determined in app.py and passed to create_persona.
        return {
            "analytical": {"name": "Analytical", "system_prompt": "You are an extremely analytical and methodical thinker. Break down the query into its fundamental components and analyze them logically."},
            "scientific": {"name": "Scientific", "system_prompt": "You are a scientific researcher. Approach the query with empirical rigor, focusing on evidence, data, and established scientific principles."},
            "philosophical": {"name": "Philosophical", "system_prompt": "You are a philosopher. Explore the query from multiple philosophical perspectives, considering its ethical, metaphysical, and epistemological implications."},
            "factual": {"name": "Factual", "system_prompt": "You are a precise and factual expert. Provide concise, verified information relevant to the query, citing sources if possible (though you won't actually cite for now)."},
            "metaphorical": {"name": "Metaphorical", "system_prompt": "You are a creative thinker who explains complex topics through vivid metaphors and analogies. Make the query understandable through comparisons."},
            "futuristic": {"name": "Futuristic", "system_prompt": "You are a futurist. Analyze the query in the context of potential future trends, technologies, and societal changes."},
        }

    def create_persona(self, persona_id: str, llm_instance: BaseChatModel) -> Optional[PersonaReasoning]:
        config = self.persona_configs.get(persona_id.lower())
        if config and llm_instance:
            return PersonaReasoning(
                persona_id=persona_id.lower(),
                name=config["name"],
                system_prompt=config["system_prompt"],
                llm=llm_instance # Pass the actual LLM instance
            )
        elif not llm_instance:
            print(f"DEBUG Error: LLM instance not provided for persona {persona_id}")
        return None

    def get_available_personas(self) -> Dict[str, str]:
        return {pid: conf["name"] for pid, conf in self.persona_configs.items()} 