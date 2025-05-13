---
title: InsightFlow AI
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
# app_port: 7860 # App port is usually not needed here if using standard Chainlit port or Docker EXPOSE
short_description: "AI research assistant: RAG, visuals, multi-perspective"
---

# InsightFlow AI: Multi-Perspective Research Assistant

InsightFlow AI is an advanced research assistant that analyzes topics from multiple perspectives, providing a comprehensive and nuanced understanding of complex subjects. It now features Retrieval Augmented Generation (RAG) for supported personas and a streamlined UI using Chainlit's chat settings.

![InsightFlow AI](https://huggingface.co/datasets/suhas/InsightFlow-AI-demo/resolve/main/insightflow_banner.png)

## Features

### Multiple Perspective Analysis
- **Analytical**: Logical examination with methodical connections and patterns
- **Scientific**: Evidence-based reasoning grounded in empirical data
- **Philosophical**: Holistic exploration of deeper meaning and implications
- **Factual**: Straightforward presentation of verified information
- **Metaphorical**: Creative explanations through vivid analogies
- **Futuristic**: Forward-looking exploration of potential developments

### Personality Perspectives
- **Sherlock Holmes**: Deductive reasoning with detailed observation
- **Richard Feynman**: First-principles physics with clear explanations
- **Hannah Fry**: Math-meets-society storytelling with practical examples

### Visualization Capabilities
- **Concept Maps**: Automatically generated Mermaid diagrams showing relationships
- **Visual Notes**: DALL-E generated hand-drawn style visualizations of key insights
- **Visual-Only Mode**: Option to focus on visual representations for faster comprehension (via Settings)

### Retrieval Augmented Generation (RAG)
- Supported personas (e.g., Analytical, Philosophical, Metaphorical) can search dedicated knowledge bases for more informed answers.
- Toggle RAG functionality via the Settings panel.

### Export Options
- **Markdown Export**: Save analyses as formatted markdown with embedded visualizations
- **PDF Export**: Generate professionally formatted PDF documents

## How to Use

1.  **Configure Settings (⚙️ icon)**: 
    *   Select a Persona Team (e.g., Balanced Overview) or toggle individual personas.
    *   Adjust other settings like RAG, Direct Mode, Quick Mode, and visibility of perspectives/visualizations.
2.  **Ask Your Question**: Type any research question or topic to analyze.
3.  **Review Insights**: Explore the synthesized view and (if enabled) individual perspectives and visualizations.
4.  **Get Help**: Type `/help` for a detailed guide on features and settings.
5.  **Export Results**: Use `/export_md` or `/export_pdf` to save your analysis (Note: Full export functionality is under development).

## Commands

Most settings are now conveniently managed via the **Settings (⚙️) panel** in the UI.
Type `/help` for a comprehensive guide to all features and settings.

```
# Core Commands
/help                  - Displays a detailed help message about features and settings.

# Mode Toggles (Also available in Settings)
/direct on|off         - Toggle direct LLM mode (bypasses multi-persona).
/perspectives on|off   - Toggle showing individual perspectives.
/visualization on|off  - Toggle showing visualizations (Mermaid & DALL-E).
/quick_mode on|off     - Toggle Quick Mode (uses a smaller, predefined set of personas).
/rag on|off            - Toggle Retrieval Augmented Generation for supported personas.

# Export Options (Functionality under development)
/export_md             - Export the current insight analysis to a markdown file.
/export_pdf            - Export the current insight analysis to a PDF file.

# Legacy Persona Commands (Advanced - Consider using Settings panel for primary persona management)
# /add [persona_name]    - Add a perspective to your research team
# /remove [persona_name] - Remove a perspective from your team
# /list                  - Show all available perspectives
# /team                  - Show your current team and settings
```

## Example Topics

- Historical events from multiple perspectives
- Scientific concepts with philosophical implications
- Societal issues that benefit from diverse viewpoints
- Future trends analyzed from different angles
- Complex problems requiring multi-faceted analysis

## Technical Details

Built with Python using:
- LangGraph for orchestration
- OpenAI APIs for reasoning and visualization
- Chainlit for the user interface
- Custom persona system for perspective management

## Try These Examples

- "The impact of artificial intelligence on society"
- "Climate change adaptation strategies"
- "Consciousness and its relationship to the brain"
- "The future of work in the next 20 years"
- "Ancient Greek philosophy and its relevance today"

## Feedback and Support

For questions, feedback, or support, please open an issue on the [GitHub repository](https://github.com/suhas/InsightFlow-AI) or comment on this Space.
