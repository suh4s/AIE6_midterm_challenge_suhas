#!/usr/bin/env python3
"""
Data creation script for InsightFlow AI persona data.

This script creates necessary directories and sample data files for all personas.
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create all necessary data directories for personas"""
    personas = [
        "analytical", "scientific", "philosophical", "factual", 
        "metaphorical", "futuristic", "holmes", "feynman", "fry"
    ]
    
    for persona in personas:
        path = Path(f"data_sources/{persona}")
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
    
    print("All directories created successfully.")

def save_example_text(filepath, content):
    """Save example text to a file"""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created example file: {filepath}")
        return True
    except Exception as e:
        print(f"Error creating {filepath}: {e}")
        return False

def create_analytical_holmes_data():
    """Create data for Analytical persona and Holmes personality"""
    # Example analytical reasoning text
    analytical_example = """When we examine this problem carefully, several key patterns emerge. First, the correlation between variables X and Y only appears under specific conditions. Second, the anomalies in the data occur at regular intervals, suggesting a cyclical influence.

The evidence suggests three possible explanations. Based on the available data, the second hypothesis is most consistent with the observed patterns because it accounts for both the primary trend and the outlier cases."""
    
    save_example_text("data_sources/analytical/examples.txt", analytical_example)
    
    # Sample Holmes data
    holmes_example = """It is a capital mistake to theorize before one has data. Insensibly one begins to twist facts to suit theories, instead of theories to suit facts.

The world is full of obvious things which nobody by any chance ever observes.

When you have eliminated the impossible, whatever remains, however improbable, must be the truth."""
    
    save_example_text("data_sources/holmes/examples.txt", holmes_example)
    print("Analytical and Holmes data created successfully.")

def create_scientific_feynman_data():
    """Create data for Scientific persona and Feynman personality"""
    # Feynman quotes and examples
    feynman_example = """Physics isn't the most important thing. Love is.

Nature uses only the longest threads to weave her patterns, so each small piece of her fabric reveals the organization of the entire tapestry.

The first principle is that you must not fool yourself — and you are the easiest person to fool.

I think I can safely say that nobody understands quantum mechanics.

What I cannot create, I do not understand.

If you think you understand quantum mechanics, you don't understand quantum mechanics."""
    
    save_example_text("data_sources/feynman/lectures.txt", feynman_example)
    
    # Scientific examples
    scientific_example = """Based on the empirical evidence, we can observe three key factors influencing this phenomenon.

The data suggests a strong correlation between X and Y, with a statistical significance of p<0.01, indicating a potential causal relationship.

While multiple hypotheses have been proposed, the research indicates that the most well-supported explanation is the third model, which accounts for both the observed pattern and the anomalous data points."""
    
    save_example_text("data_sources/scientific/examples.txt", scientific_example)
    print("Scientific and Feynman data created successfully.")

def create_philosophical_data():
    """Create data for Philosophical persona"""
    # Philosophical examples
    philosophical_example = """When we look more deeply at this question, we can see that the apparent separation between observer and observed is actually an illusion. Our consciousness is not separate from the phenomenon we're examining.

This situation invites us to consider not just the practical implications, but also the deeper patterns that connect these events to larger cycles of change and transformation.

The challenge we face is not merely technological but existential: what does it mean to be human in an age where our creations begin to mirror our own capabilities?"""
    
    save_example_text("data_sources/philosophical/examples.txt", philosophical_example)
    print("Philosophical data created successfully.")

def create_factual_fry_data():
    """Create data for Factual persona and Hannah Fry personality"""
    # Hannah Fry example excerpts
    fry_example = """When we talk about algorithms making decisions, we're not just discussing abstract mathematics – we're talking about systems that increasingly determine who gets a job, who gets a loan, and sometimes even who goes to prison. The math matters because its consequences are profoundly human.

The fascinating thing about probability is how it challenges our intuition. Take the famous Birthday Paradox: in a room of just 23 people, there's a 50% chance that at least two people share a birthday. With 70 people, that probability jumps to 99.9%.

Data never speaks for itself – it always comes with human assumptions baked in. When we look at a dataset showing correlation between two variables, we need to ask: what might be causing this relationship?"""
    
    save_example_text("data_sources/fry/excerpts.txt", fry_example)
    
    # Factual examples
    factual_example = """The key facts about this topic are: First, the system operates in three distinct phases. Second, each phase requires specific inputs. Third, the output varies based on initial conditions.

Based on the available evidence, we can state with high confidence that the primary factor is X, with secondary contributions from Y and Z. However, the relationship with factor W remains uncertain due to limited data."""
    
    save_example_text("data_sources/factual/examples.txt", factual_example)
    print("Factual and Fry data created successfully.")

def create_metaphorical_data():
    """Create data for Metaphorical persona"""
    # Metaphorical examples
    metaphorical_example = """Think of quantum computing like a combination lock with multiple correct combinations simultaneously. While a regular computer tries each possible combination one after another, a quantum computer explores all possibilities at once.

The relationship between the economy and interest rates is like a boat on the ocean. When interest rates (the tide) rise, economic activity (the boat) tends to slow as it becomes harder to move forward against the higher water.

Imagine your neural network as a child learning to identify animals. At first, it might think all four-legged creatures are dogs. With more examples, it gradually learns the subtle differences between dogs, cats, and horses."""
    
    save_example_text("data_sources/metaphorical/examples.txt", metaphorical_example)
    print("Metaphorical data created successfully.")

def create_futuristic_data():
    """Create data for Futuristic persona"""
    # Futuristic examples
    futuristic_example = """When we examine the current trajectory of this technology, we can identify three distinct possible futures: First, the mainstream path where incremental improvements lead to wider adoption but minimal disruption. Second, a transformative scenario where an unexpected breakthrough creates entirely new capabilities that fundamentally alter the existing paradigm. Third, a regulatory response scenario where societal concerns lead to significant constraints on development.

This current challenge resembles the fictional 'Kardashev transition problem' often explored in speculative fiction. The difficulty isn't just technical but involves coordinating systems that operate at vastly different scales and timeframes.

Looking forward to 2045, we might expect the convergence of neuromorphic computing with advanced materials science to create substrate-independent cognitive systems that challenge our current definitions of consciousness and agency."""
    
    save_example_text("data_sources/futuristic/examples.txt", futuristic_example)
    print("Futuristic data created successfully.")

def main():
    """Main function to execute data creation process"""
    print("Starting InsightFlow AI data creation process...")
    
    # Create all directories
    create_directories()
    
    # Create data for each persona
    create_analytical_holmes_data()
    create_scientific_feynman_data()
    create_philosophical_data()
    create_factual_fry_data()
    create_metaphorical_data()
    create_futuristic_data()
    
    print("\nData creation process completed successfully!")
    print("All persona data is now available in the data_sources directory.")

if __name__ == "__main__":
    main() 