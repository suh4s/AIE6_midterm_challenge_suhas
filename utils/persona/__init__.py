"""
Persona system for InsightFlow AI.

This module provides the two-tier persona system with:
1. Base persona types (Analytical, Scientific, Philosophical, etc.)
2. Specific personalities (Holmes, Feynman, Fry)
"""

# Import key classes for easier access
from utils.persona.base import PersonaReasoning, PersonaFactory
from utils.persona.impl import (
    LLMPersonaReasoning,
    AnalyticalReasoning,
    ScientificReasoning,
    PhilosophicalReasoning,
    FactualReasoning, 
    MetaphoricalReasoning,
    FuturisticReasoning,
    HolmesReasoning,
    FeynmanReasoning,
    FryReasoning
) 