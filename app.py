# InsightFlow AI - Main Application
import chainlit as cl
from insight_state import InsightFlowState # Import InsightFlowState
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI # Import AsyncOpenAI for DALL-E
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage # Added for prompts
from utils.persona import PersonaFactory # Import PersonaFactory
from utils.visualization_utils import generate_dalle_image, generate_mermaid_code # <--- UPDATED IMPORT
import asyncio # For asyncio.gather in execute_persona_tasks
from langchain_core.callbacks.base import AsyncCallbackHandler # <--- ADDED for progress
from typing import Any, Dict, List, Optional, Union # <--- ADDED for callbacks
from langchain_core.outputs import LLMResult # <--- ADDED for callbacks
import datetime # For export filenames
from pathlib import Path # For export directory
from chainlit.input_widget import Switch, Select # <--- REMOVE Option import
# from chainlit.input_widget import Collapse # <--- COMMENT OUT FOR NOW
# from chainlit.element import Divider # <--- REMOVE Collapse from this import for now

# --- RAG UTILS IMPORT ---
from utils.rag_utils import get_embedding_model_instance, get_relevant_context_for_query
# --- END RAG UTILS IMPORT ---

# --- GLOBAL CONFIGURATION STATE ---
_configurations_initialized = False
# _embedding_model_initialized = False # REMOVE OLD GLOBAL FLAG
llm_planner = None
llm_synthesizer = None
llm_direct = None
llm_analytical = None
llm_scientific = None
llm_philosophical = None
llm_factual = None
llm_metaphorical = None
llm_futuristic = None
llm_mermaid_generator = None # <--- ADDED
openai_async_client = None # For DALL-E
PERSONA_LLM_MAP = {}

# Embedding Model Identifiers
OPENAI_EMBED_MODEL_ID = "text-embedding-3-small"
# IMPORTANT: Replace with your actual fine-tuned model ID from Hugging Face Hub after training
FINETUNED_BALANCED_TEAM_EMBED_ID = "suh4s/insightflow-balanced-team-embed-v1-7099e82c-e4c8-48ed-88a8-36bd9255036b" # Placeholder

QUICK_MODE_PERSONAS = ["analytical", "factual"] # Default personas for Quick Mode
# --- RAG Configuration (moved to global scope) ---
RAG_ENABLED_PERSONA_IDS = ["analytical", "philosophical", "metaphorical"]
# --- End RAG Configuration ---

PERSONA_TEAMS = {
    "creative_synthesis": {
        "name": "🎨 Creative Synthesis Team",
        "description": "Generates novel ideas and artistic interpretations.",
        "members": ["metaphorical", "futuristic", "philosophical"]
    },
    "data_driven_analysis": {
        "name": "📊 Data-Driven Analysis Squad",
        "description": "Focuses on factual accuracy and logical deduction.",
        "members": ["analytical", "factual", "scientific"]
    },
    "balanced_overview": {
        "name": "⚖️ Balanced Overview Group",
        "description": "Provides a well-rounded perspective.",
        "members": ["analytical", "philosophical", "metaphorical"] # UPDATED: factual -> metaphorical
    }
}

def initialize_configurations():
    """Loads environment variables and initializes LLM configurations."""
    global _configurations_initialized
    global llm_planner, llm_synthesizer, llm_direct, llm_analytical, llm_scientific
    global llm_philosophical, llm_factual, llm_metaphorical, llm_futuristic
    global llm_mermaid_generator # <--- ADDED
    global openai_async_client # Add new client to globals
    global PERSONA_LLM_MAP

    if _configurations_initialized:
        return

    print("Initializing configurations: Loading .env and setting up LLMs...")
    load_dotenv()

    # LLM CONFIGURATIONS
    llm_planner = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    llm_synthesizer = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    llm_direct = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    llm_analytical = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    llm_scientific = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    llm_philosophical = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    llm_factual = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    llm_metaphorical = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
    llm_futuristic = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
    llm_mermaid_generator = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1) # <--- ADDED INITIALIZATION

    # Initialize OpenAI client for DALL-E etc.
    openai_async_client = AsyncOpenAI()

    # Mapping persona IDs to their specific LLM instances
    PERSONA_LLM_MAP.update({
        "analytical": llm_analytical,
        "scientific": llm_scientific,
        "philosophical": llm_philosophical,
        "factual": llm_factual,
        "metaphorical": llm_metaphorical,
        "futuristic": llm_futuristic,
    })
    
    _configurations_initialized = True
    print("Configurations initialized.")

# Load environment variables first
# load_dotenv() # Moved to initialize_configurations

# --- LLM CONFIGURATIONS ---
# Configurations based on tests/test_llm_config.py
# llm_planner = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1) # Moved
# llm_synthesizer = ChatOpenAI(model="gpt-4o-mini", temperature=0.4) # Moved
# llm_direct = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3) # Moved
# llm_analytical = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2) # Moved
# llm_scientific = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3) # Moved
# llm_philosophical = ChatOpenAI(model="gpt-4o-mini", temperature=0.5) # Moved
# llm_factual = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1) # Moved
# llm_metaphorical = ChatOpenAI(model="gpt-4o-mini", temperature=0.6) # Moved
# llm_futuristic = ChatOpenAI(model="gpt-4o-mini", temperature=0.6) # Moved

# Mapping persona IDs to their specific LLM instances
# PERSONA_LLM_MAP = { # Moved and will be populated in initialize_configurations
#     "analytical": llm_analytical,
#     "scientific": llm_scientific,
#     "philosophical": llm_philosophical,
#     "factual": llm_factual,
#     "metaphorical": llm_metaphorical,
#     "futuristic": llm_futuristic,
    # Add other personas here if they have dedicated LLMs or share one from above
# }

# --- SYSTEM PROMPTS (from original app.py) ---
DIRECT_SYSPROMPT = """You are a highly intelligent AI assistant that provides clear, direct, and helpful answers.
Your responses should be accurate, concise, and well-reasoned."""

SYNTHESIZER_SYSTEM_PROMPT_TEMPLATE = """You are a master synthesizer AI. Your task is to integrate the following diverse perspectives into a single, coherent, and insightful response. Ensure that the final synthesis is well-structured, easy to understand, and accurately reflects the nuances of each provided viewpoint. Do not simply list the perspectives; weave them together.

Perspectives:
{formatted_perspectives}

Synthesized Response:"""

# --- LANGGRAPH NODE FUNCTIONS (DUMMIES FOR NOW) ---
async def run_planner_agent(state: InsightFlowState) -> InsightFlowState:
    print(f"Node: run_planner_agent, Query: {state.get('query')}")
    progress_msg = cl.user_session.get("progress_msg")
    completed_steps_log = cl.user_session.get("completed_steps_log")
    
    activity_description = "Planning research approach..."
    activity_emoji = "📅"
    current_activity_display = f"{activity_emoji} {activity_description} (10%)"

    if progress_msg:
        progress_msg.content = f"**Current Activity:**\n{current_activity_display}"
        if completed_steps_log:
            progress_msg.content = f"**Completed Steps:**\n{(chr(10)).join(completed_steps_log)}\n\n{progress_msg.content}"
        await progress_msg.update()
    
    completed_steps_log.append(f"{activity_emoji} {activity_description}") # Log without percentage
    cl.user_session.set("completed_steps_log", completed_steps_log)

    state["current_step_name"] = "execute_persona_tasks"
    return state

async def execute_persona_tasks(state: InsightFlowState) -> InsightFlowState:
    print(f"Node: execute_persona_tasks")
    progress_msg = cl.user_session.get("progress_msg")
    completed_steps_log = cl.user_session.get("completed_steps_log", [])
    
    activity_description = "Generating perspectives from selected team..."
    activity_emoji = "🧠"
    current_activity_display = f"{activity_emoji} {activity_description} (20%)"

    if progress_msg:
        progress_msg.content = f"**Completed Steps:**\n{(chr(10)).join(completed_steps_log)}\n\n**Current Activity:**\n{current_activity_display}"
        await progress_msg.update()

    persona_factory: PersonaFactory = cl.user_session.get("persona_factory")
    selected_persona_ids = state.get("selected_personas", [])
    query = state.get("query")
    
    if not selected_persona_ids:
        state["persona_responses"] = {"error": "No personas selected for perspective generation."}
        state["current_step_name"] = "synthesize_responses" # Or an error handling state
        if progress_msg: completed_steps_log.append(f"{activity_emoji} {activity_description} - No personas selected.")
        cl.user_session.set("completed_steps_log", completed_steps_log)
        return state

    await cl.Message(content=f"Invoking {len(selected_persona_ids)} personas...").send()

    tasks = []
    # --- RAG Enhancement ---
    embedding_model = cl.user_session.get("embedding_model_instance") # <--- GET MODEL FROM SESSION
    global_rag_enabled = cl.user_session.get("enable_rag", True) 
    # --- End RAG Enhancement ---

    valid_persona_ids_for_results = [] # Keep track of personas for which tasks were created
    for persona_id in selected_persona_ids:
        persona_llm = PERSONA_LLM_MAP.get(persona_id.lower())
        if not persona_llm:
            print(f"Warning: LLM not found for persona {persona_id}. Skipping.")
            continue

        persona = persona_factory.create_persona(persona_id, persona_llm)
        if persona:
            final_query_for_llm = query # Default to original query
            
            # --- RAG Integration for Balanced Team Personas ---
            if global_rag_enabled and persona.persona_id.lower() in RAG_ENABLED_PERSONA_IDS: # Check global toggle first; Changed persona.id to persona.persona_id
                if embedding_model:
                    rag_progress_message = f"\n   🔍 Persona '{persona.name}' (RAG enabled): Searching knowledge base for context related to: '{query[:50]}...'"
                    if progress_msg: await progress_msg.stream_token(rag_progress_message)
                    print(rag_progress_message.strip())
                    
                    retrieved_context = await get_relevant_context_for_query(query, persona.persona_id, embedding_model)
                    
                    if retrieved_context:
                        context_log_msg = f"   ✅ Context retrieved for '{persona.name}'. Augmenting prompt."
                        # print(f"Retrieved context for {persona.id}:\n{retrieved_context}") # Full context - verbose
                        final_query_for_llm = f"""As the {persona.name}, consider the following retrieved context to answer the user's query.
Integrate this context with your inherent expertise and perspective.

Retrieved Context:
---
{retrieved_context}
---

User Query: {query}

Answer as {persona.name}:
"""
                        if progress_msg: await progress_msg.stream_token(f"\n   💡 Context found for {persona.name}. Crafting response...")
                        print(context_log_msg)
                    else:
                        no_context_log_msg = f"   ℹ️ No specific context found for '{persona.name}' for this query. Relying on general knowledge."
                        if progress_msg: await progress_msg.stream_token(f"\n   🧐 No specific context found for {persona.name}. Proceeding with general knowledge...")
                        print(no_context_log_msg)
                        # Prompt when no context is found - persona relies on its base system prompt and expertise
                        # The persona.generate_perspective method will use its inherent system_prompt.
                        # We can just pass the original query, or a slightly modified one indicating no specific context was found.
                        final_query_for_llm = f"""As the {persona.name}, answer the user's query using your inherent expertise and perspective.
No specific context from the knowledge base was retrieved for this particular query.

User Query: {query}

Answer as {persona.name}:
"""
                else:
                    no_embed_msg = f"\n   ⚠️ Embedding model (current selection) not available for RAG for persona '{persona.name}'. Using general knowledge."
                    if progress_msg: await progress_msg.stream_token(no_embed_msg)
                    print(no_embed_msg.strip())
            # --- End RAG Integration ---
            
            # Stream persona-specific LLM call message
            llm_call_msg = f"\n   🗣️ Consulting AI for {persona.name} perspective..."
            if progress_msg: await progress_msg.stream_token(llm_call_msg)
            print(llm_call_msg.strip())
            
            tasks.append(persona.generate_perspective(final_query_for_llm)) # Use final_query_for_llm
            valid_persona_ids_for_results.append(persona_id)
        else:
            print(f"Warning: Could not create persona object for {persona_id}. Skipping.")
    
    state["persona_responses"] = {} # Initialize/clear previous responses
    if tasks:
        try:
            # Timeout logic can be added here if needed for asyncio.gather
            if progress_msg: await progress_msg.stream_token(f"\n   ↪ Invoking {len(valid_persona_ids_for_results)} personas...")
            persona_results = await asyncio.gather(*tasks)
            # Store responses keyed by the valid persona_ids used for tasks
            for i, persona_id in enumerate(valid_persona_ids_for_results):
                state["persona_responses"][persona_id] = persona_results[i]
        except Exception as e:
            print(f"Error during persona perspective generation: {e}")
            state["error_message"] = f"Error generating perspectives: {str(e)[:100]}"
            # Optionally, populate partial results if some tasks succeeded before error
            # For now, just reports error. Individual task errors could be handled in PersonaReasoning too.

    if progress_msg: 
        completed_activity_description = "Perspectives generated."
        # Emoji for this completed step was 🧠 from current_activity_display
        completed_steps_log.append(f"{activity_emoji} {completed_activity_description}")
        cl.user_session.set("completed_steps_log", completed_steps_log)
        
        # Display updated completed list; the next node will set the new "Current Activity"
        # For this brief moment, show only completed steps before next node updates with its current activity
        progress_msg.content = f"**Completed Steps:**\n{(chr(10)).join(completed_steps_log)}\n\n**Current Activity:**\n✅ Perspectives generated. (60%)" # Show this as current before next node takes over
        await progress_msg.update()

    state["current_step_name"] = "synthesize_responses"
    return state

async def synthesize_responses(state: InsightFlowState) -> InsightFlowState:
    print(f"Node: synthesize_responses")
    progress_msg = cl.user_session.get("progress_msg")
    completed_steps_log = cl.user_session.get("completed_steps_log")
    
    activity_description = "Synthesizing insights..."
    activity_emoji = "✍️"
    current_activity_display = f"{activity_emoji} {activity_description} (65%)"

    if progress_msg: 
        progress_msg.content = f"**Current Activity:**\n{current_activity_display}"
        if completed_steps_log:
            progress_msg.content = f"**Completed Steps:**\n{(chr(10)).join(completed_steps_log)}\n\n{progress_msg.content}"
        await progress_msg.update()
    
    persona_responses = state.get("persona_responses", {})
    
    if not persona_responses:
        print("No persona responses to synthesize.")
        state["synthesized_response"] = "No perspectives were available to synthesize."
        state["current_step_name"] = "generate_visualization"
        return state

    formatted_perspectives_list = []
    for persona_id, response_text in persona_responses.items():
        formatted_perspectives_list.append(f"- Perspective from {persona_id}: {response_text}")
    
    formatted_perspectives_string = "\n".join(formatted_perspectives_list)
    
    final_prompt_content = SYNTHESIZER_SYSTEM_PROMPT_TEMPLATE.format(
        formatted_perspectives=formatted_perspectives_string
    )
    
    messages = [
        SystemMessage(content=final_prompt_content)
    ]
    
    try:
        # Ensure llm_synthesizer is available (initialized by initialize_configurations)
        if llm_synthesizer is None:
            print("Error: llm_synthesizer is not initialized.")
            state["error_message"] = "Synthesizer LLM not available."
            state["synthesized_response"] = "Synthesis failed due to internal error."
            state["current_step_name"] = "error_presenting" # Or a suitable error state
            return state

        ai_response = await llm_synthesizer.ainvoke(messages)
        synthesized_text = ai_response.content
        state["synthesized_response"] = synthesized_text
        print(f"Synthesized response: {synthesized_text[:200]}...") # Log snippet
    except Exception as e:
        print(f"Error during synthesis: {e}")
        state["error_message"] = f"Synthesis error: {str(e)[:100]}"
        state["synthesized_response"] = "Synthesis failed."
        # Optionally, decide if we proceed to visualization or an error state
        # For now, let's assume we still try to visualize if there's a partial/failed synthesis

    if progress_msg: 
        completed_activity_description = "Insights synthesized."
        completed_steps_log.append(f"{activity_emoji} {completed_activity_description}")
        cl.user_session.set("completed_steps_log", completed_steps_log)
        progress_msg.content = f"**Completed Steps:**\n{(chr(10)).join(completed_steps_log)}\n\n**Current Activity:**\n✅ Insights synthesized. (80%)"
        await progress_msg.update()

    state["current_step_name"] = "generate_visualization"
    return state

async def generate_visualization(state: InsightFlowState) -> InsightFlowState:
    print(f"Node: generate_visualization")
    progress_msg = cl.user_session.get("progress_msg")
    completed_steps_log = cl.user_session.get("completed_steps_log")
    
    activity_description = "Creating visualizations..."
    activity_emoji = "🎨"
    current_activity_display = f"{activity_emoji} {activity_description} (85%)"

    if progress_msg: 
        progress_msg.content = f"**Current Activity:**\n{current_activity_display}"
        if completed_steps_log:
            progress_msg.content = f"**Completed Steps:**\n{(chr(10)).join(completed_steps_log)}\n\n{progress_msg.content}"
        await progress_msg.update()

    synthesized_response = state.get("synthesized_response")
    image_url = None
    mermaid_code_output = None # Changed variable name for clarity

    # DALL-E Image Generation (existing logic)
    if synthesized_response and openai_async_client:
        # Log the full DALL-E prompt for debugging if issues persist
        # print(f"Full DALL-E prompt for visualization: {dalle_prompt}") 
        dalle_prompt = f"A hand-drawn style visual note or sketch representing the key concepts of: {synthesized_response}"
        if len(dalle_prompt) > 4000: 
            dalle_prompt = dalle_prompt[:3997] + "..."
        
        print(f"Attempting DALL-E image generation for: {dalle_prompt[:100]}...")
        image_url = await generate_dalle_image(prompt=dalle_prompt, client=openai_async_client)
        if image_url:
            state["visualization_image_url"] = image_url
            print(f"DALL-E Image URL: {image_url}")
        else:
            print("DALL-E image generation failed or returned no URL.")
            state["visualization_image_url"] = None 
    elif not synthesized_response:
        print("No synthesized response available to generate DALL-E image.")
        state["visualization_image_url"] = None
    elif not openai_async_client:
        print("OpenAI async client not initialized, skipping DALL-E generation.")
        state["visualization_image_url"] = None
    
    # Mermaid Code Generation
    if synthesized_response and llm_mermaid_generator: # Check if both are available
        print(f"Attempting Mermaid code generation for: {synthesized_response[:100]}...")
        mermaid_code_output = await generate_mermaid_code(synthesized_response, llm_mermaid_generator)
        if mermaid_code_output:
            state["visualization_code"] = mermaid_code_output
            print(f"Mermaid code generated: {mermaid_code_output[:100]}...")
        else:
            print("Mermaid code generation failed or returned no code.")
            state["visualization_code"] = None # Ensure it's None if failed
    else:
        print("Skipping Mermaid code generation due to missing response or LLM.")
        state["visualization_code"] = None

    if progress_msg: 
        completed_activity_description = "Visualizations created."
        completed_steps_log.append(f"{activity_emoji} {completed_activity_description}")
        cl.user_session.set("completed_steps_log", completed_steps_log)
        progress_msg.content = f"**Completed Steps:**\n{(chr(10)).join(completed_steps_log)}\n\n**Current Activity:**\n✅ Visualizations created. (95%)"
        await progress_msg.update()

    state["current_step_name"] = "present_results"
    return state

async def present_results(state: InsightFlowState) -> InsightFlowState:
    print(f"Node: present_results")
    progress_msg = cl.user_session.get("progress_msg")
    completed_steps_log = cl.user_session.get("completed_steps_log")
    
    activity_description = "Preparing final presentation..."
    activity_emoji = "🎁"
    current_activity_display = f"{activity_emoji} {activity_description} (98%)"

    if progress_msg: 
        progress_msg.content = f"**Current Activity:**\n{current_activity_display}"
        if completed_steps_log:
            progress_msg.content = f"**Completed Steps:**\n{(chr(10)).join(completed_steps_log)}\n\n{progress_msg.content}"
        await progress_msg.update()

    show_visualization = cl.user_session.get("show_visualization", True)
    show_perspectives = cl.user_session.get("show_perspectives", True)
    synthesized_response = state.get("synthesized_response")

    # 1. Send Synthesized Response (always)
    if synthesized_response:
        await cl.Message(content=synthesized_response, author="Synthesized Insight").send()
    else:
        await cl.Message(content="No synthesized response was generated.", author="System").send()

    # 2. Send Visualizations (if enabled and available)
    if show_visualization:
        image_url = state.get("visualization_image_url")
        if image_url:
            image_element = cl.Image(
                url=image_url, 
                name="dalle_visualization", 
                display="inline", 
                size="large"
            )
            # Send image with a title in the content or as a separate message
            await cl.Message(content="Visual Summary:", elements=[image_element], author="System").send()
        
        mermaid_code = state.get("visualization_code")
        if mermaid_code:
            mermaid_element = cl.Text(
                content=mermaid_code, 
                mime_type="text/mermaid",
                name="generated_diagram", 
                display="inline"
            )
            await cl.Message(content="Concept Map:", elements=[mermaid_element], author="System").send()

    # 3. Send Persona Perspectives within cl.Collapse (if enabled and available)
    if show_perspectives:
        persona_responses = state.get("persona_responses")
        if persona_responses:
            perspective_elements = []
            for persona_id, response_text in persona_responses.items():
                if response_text:
                    # Temporarily revert to sending simple messages for perspectives
                    await cl.Message(content=f"**Perspective from {persona_id.capitalize()}:**\n{response_text}", author=persona_id.capitalize()).send()
                    # perspective_elements.append(
                    #     Collapse( # <--- COMMENT OUT USAGE
                    #         label=f"👁️ Perspective from {persona_id.capitalize()}", 
                    #         content=response_text,
                    #         initial_collapsed=True # Start collapsed
                    #     )
                    # )
            # if perspective_elements:
            #     await cl.Message(
            #         content="Dive Deeper into Individual Perspectives:", 
            #         elements=perspective_elements, 
            #         author="System"
            #     ).send()
    
    state["current_step_name"] = "results_presented"
    
    if progress_msg: 
        # Log the "Preparing presentation" step as completed first
        completed_steps_log.append(f"{activity_emoji} {activity_description}") 
        cl.user_session.set("completed_steps_log", completed_steps_log)

        # Show the 100% completion message as current activity initially
        final_emoji = "✨"
        final_message_description = "All insights presented!"
        current_activity_display = f"{final_emoji} {final_message_description} (100%)"
        
        progress_msg.content = f"**Completed Steps:**\n{(chr(10)).join(completed_steps_log)}\n\n**Current Activity:**\n{current_activity_display}"
        await progress_msg.update()

        # Now, move the 100% step to completed and finalize the progress message
        await cl.sleep(0.5) # Optional short delay for UI to update before final change
        completed_steps_log.append(f"{final_emoji} {final_message_description}")
        cl.user_session.set("completed_steps_log", completed_steps_log)
        
        # Generate witty remark
        query_for_remark = state.get("query", "this topic")
        personas_for_remark = state.get("selected_personas", [])
        snippet_for_remark = state.get("synthesized_response")
        witty_remark = await generate_witty_completion_remark(query_for_remark, personas_for_remark, snippet_for_remark)

        progress_msg.content = f"**Completed Steps:**\n{(chr(10)).join(completed_steps_log)}\n\n{witty_remark}"
        await progress_msg.update()

    # Send a final message that all content is loaded, then actions
    await cl.Message(content="All insights presented. You can now export the results.").send()

    # Add Export Actions
    export_actions = [
        cl.Action(name="export_markdown", label="Export to Markdown", value="markdown", description="Export results to a Markdown file.", payload={}),
        cl.Action(name="export_pdf", label="Export to PDF", value="pdf", description="Export results to a PDF file.", payload={})
    ]
    await cl.Message(content="", actions=export_actions).send()

    return state

# --- LANGGRAPH SETUP ---
insight_graph_builder = StateGraph(InsightFlowState)

# Add nodes
insight_graph_builder.add_node("planner_agent", run_planner_agent)
insight_graph_builder.add_node("execute_persona_tasks", execute_persona_tasks)
insight_graph_builder.add_node("synthesize_responses", synthesize_responses)
insight_graph_builder.add_node("generate_visualization", generate_visualization)
insight_graph_builder.add_node("present_results", present_results)

# Set entry point
insight_graph_builder.set_entry_point("planner_agent")

# Add edges
insight_graph_builder.add_edge("planner_agent", "execute_persona_tasks")
insight_graph_builder.add_edge("execute_persona_tasks", "synthesize_responses")
insight_graph_builder.add_edge("synthesize_responses", "generate_visualization")
insight_graph_builder.add_edge("generate_visualization", "present_results")
insight_graph_builder.add_edge("present_results", END)

# Compile the graph
insight_flow_graph = insight_graph_builder.compile()

print("LangGraph setup complete.")

# --- CUSTOM CALLBACK HANDLER FOR PROGRESS UPDATES --- #
class InsightFlowCallbackHandler(AsyncCallbackHandler):
    def __init__(self, progress_message: cl.Message):
        self.progress_message = progress_message
        self.step_counter = 0

    async def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running. Now simplified as nodes will provide more specific updates."""
        # This callback might still be useful for very generic LangGraph level start/stop
        # but specific node progress will be handled within the node functions themselves.
        # Avoid printing "Unknown chain/node" if possible.
        # langgraph_event_name = serialized.get("name") or (serialized.get("id")[-1] if isinstance(serialized.get("id"), list) and serialized.get("id") else None)
        # if self.progress_message and langgraph_event_name and not langgraph_event_name.startswith("__"):
        #     self.step_counter += 1
        #     await self.progress_message.stream_token(f"\n⏳ Step {self.step_counter}: Processing {langgraph_event_name}...\")
        pass # Node functions will now handle more granular progress updates

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running. This will now be more generic or can be removed if node-specific messages are enough."""
        # llm_display_name = "LLM" # Sensible default
        # if serialized:
        #     id_list = serialized.get("id")
        #     if isinstance(id_list, list) and len(id_list) > 0:
        #         raw_name_part = id_list[0] 
        #         if isinstance(raw_name_part, str):
        #             if "ChatOpenAI".lower() in raw_name_part.lower():
        #                 llm_display_name = "OpenAI Model"
        #             elif "LLM" in raw_name_part.upper():
        #                 llm_display_name = raw_name_part
            
        #     name_val = serialized.get("name")
        #     if isinstance(name_val, str) and name_val != "Unknown LLM":
        #         if llm_display_name == "LLM" or len(name_val) > len(llm_display_name):
        #             llm_display_name = name_val
        
        # update_text = f"🗣️ Consulting {llm_display_name}... " 
        # if self.progress_message:
        #     # print(f"DEBUG on_llm_start serialized: {serialized}") 
        #     await self.progress_message.stream_token(f"\n   L {update_text}")
        pass # Specific LLM call messages are now sent from execute_persona_tasks

    # We can add on_agent_action, on_tool_start for more granular updates if nodes use agents/tools
    # For now, on_chain_start (which LangGraph nodes trigger) and on_llm_start should give good visibility.

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        # chain_name = kwargs.get("name", "a process") # LangGraph provides name in kwargs for on_chain_end
        # if self.progress_message:
        #     await self.progress_message.stream_token(f"\nCompleted: {chain_name}.\")
        pass # Node functions will provide end-of-node context if needed

# This allows the test to import these names from app if needed
__all__ = [
    "InsightFlowState", "on_chat_start", "on_message", 
    "invoke_direct_llm", "invoke_langgraph",
    "run_planner_agent", "execute_persona_tasks", "synthesize_responses",
    "generate_visualization", "present_results",
    "StateGraph", # Expose StateGraph if tests patch app.StateGraph directly
    "initialize_configurations", # Expose for testing
    "on_export_markdown", "on_export_pdf" # Add new export handlers
]

# --- CHAT SETTINGS HELPER ---
async def _apply_chat_settings_to_state():
    """Reads chat settings and updates session and insight_flow_state."""
    chat_settings_values = cl.user_session.get("chat_settings", {})
    insight_flow_state: InsightFlowState = cl.user_session.get("insight_flow_state")
    persona_factory: PersonaFactory = cl.user_session.get("persona_factory")

    if not insight_flow_state or not persona_factory:
        # Should not happen if on_chat_start ran correctly
        print("Error: State or persona factory not found while applying chat settings.")
        return

    # Update modes in user_session
    cl.user_session.set("direct_mode", chat_settings_values.get("direct_mode", False))
    cl.user_session.set("quick_mode", chat_settings_values.get("quick_mode", False))
    cl.user_session.set("show_perspectives", chat_settings_values.get("show_perspectives", True))
    cl.user_session.set("show_visualization", chat_settings_values.get("show_visualization", True))
    cl.user_session.set("enable_rag", chat_settings_values.get("enable_rag", True)) # Read the RAG toggle

    # Embedding model toggle
    new_use_finetuned_setting = chat_settings_values.get("use_finetuned_embedding", False)
    current_use_finetuned_setting = cl.user_session.get("use_finetuned_embedding", False)
    cl.user_session.set("use_finetuned_embedding", new_use_finetuned_setting)

    if new_use_finetuned_setting != current_use_finetuned_setting:
        print("Embedding model setting changed. Re-initializing model.")
        await _initialize_embedding_model_in_session() # Re-initialize if toggle changed
    
    # Update selected_personas in insight_flow_state
    selected_personas_from_settings = []
    available_persona_ids = persona_factory.get_available_personas().keys()
    for persona_id in available_persona_ids:
        if chat_settings_values.get(f"persona_{persona_id}", False):
            selected_personas_from_settings.append(persona_id)
    
    # Check if a team is selected and override individual selections
    selected_team_id = chat_settings_values.get("selected_team")
    if selected_team_id and selected_team_id != "none":
        team_info = PERSONA_TEAMS.get(selected_team_id)
        if team_info and "members" in team_info:
            selected_personas_from_settings = list(team_info["members"]) # Use team members
            print(f"Team '{team_info['name']}' selected, overriding individual personas.")
    
    mutable_state = insight_flow_state.copy()
    mutable_state["selected_personas"] = selected_personas_from_settings
    cl.user_session.set("insight_flow_state", mutable_state)
    
    print(f"Applied chat settings: Direct Mode: {cl.user_session.get('direct_mode')}, Quick Mode: {cl.user_session.get('quick_mode')}")
    print(f"Selected Personas from settings: {selected_personas_from_settings}")

@cl.on_chat_start
async def on_chat_start():
    """Initializes session state, chat settings, and sends a welcome message."""
    # global _configurations_initialized # No longer need _embedding_model_initialized here
    # The global keyword is only needed if you assign to the global variable in this scope.
    # We are only reading _configurations_initialized here.

    if not _configurations_initialized:
        initialize_configurations()

    # Initialize with default embedding model (OpenAI) first
    # This sets "use_finetuned_embedding" to False in session if not already there
    await _initialize_embedding_model_in_session(force_openai=True) 

    persona_factory = PersonaFactory()
    cl.user_session.set("persona_factory", persona_factory)

    # Default selections for personas/teams
    default_team_id = "balanced_overview" 
    default_selected_personas = list(PERSONA_TEAMS[default_team_id]["members"])

    # Default UI toggles (excluding embedding toggle, handled by _initialize_embedding_model_in_session)
    default_direct_mode = False
    default_quick_mode = False
    default_show_perspectives = True
    default_show_visualization = True
    default_enable_rag = True # Default RAG to ON
    # default_use_finetuned_embedding is set by _initialize_embedding_model_in_session(force_openai=True)

    initial_state = InsightFlowState(
        panel_type="research",
        query="",
        selected_personas=default_selected_personas,
        persona_responses={},
        synthesized_response=None,
        visualization_code=None,
        visualization_image_url=None,
        current_step_name="awaiting_query",
        error_message=None
    )
    cl.user_session.set("insight_flow_state", initial_state)

    cl.user_session.set("direct_mode", default_direct_mode)
    cl.user_session.set("quick_mode", default_quick_mode)
    cl.user_session.set("show_perspectives", default_show_perspectives)
    cl.user_session.set("show_visualization", default_show_visualization)
    cl.user_session.set("enable_rag", default_enable_rag)

    settings_inputs = []
    
    settings_inputs.append(
        Switch(id="enable_rag", label="⚙️ Enable RAG Features (for supported personas)", initial=default_enable_rag)
    )
    settings_inputs.append(
        Switch(id="use_finetuned_embedding", 
               label="🔬 Use Fine-tuned Embedding (Balanced Team - if available)", 
               initial=cl.user_session.get("use_finetuned_embedding", False)) # Initial from session
    )
    
    team_items = {"-- Select a Team (Optional) --": "none"}
    for team_id, team_info in PERSONA_TEAMS.items():
        team_items[team_info["name"]] = team_id
    
    settings_inputs.append(
        Select( 
            id="selected_team",
            label="🎯 Persona Team (Overrides individual toggles for processing)", # Corrected quotes
            items=team_items, 
            initial_value=default_team_id # This should be fine as it's a variable
        )
    )
    # settings_inputs.append(cl.Divider()) # Keep commented for now

    for persona_id, persona_config in persona_factory.persona_configs.items():
        settings_inputs.append(
            Switch(
                id=f"persona_{persona_id}", 
                label=persona_config["name"], 
                initial=persona_id in default_selected_personas
            )
        )
    
    settings_inputs.extend([
        Switch(id="direct_mode", label="🚀 Direct Mode (Quick, single LLM answers)", initial=default_direct_mode),
        Switch(id="quick_mode", label="⚡ Quick Mode (Uses max 2 personas)", initial=default_quick_mode),
        Switch(id="show_perspectives", label="👁️ Show Individual Perspectives", initial=default_show_perspectives),
        Switch(id="show_visualization", label="🎨 Show Visualizations (DALL-E & Mermaid)", initial=default_show_visualization)
    ])
    
    await cl.ChatSettings(inputs=settings_inputs).send()
    
    # New Welcome Message
    welcome_message_content = "🚀 **Welcome to InsightFlow AI!** Dive deep into any topic with a symphony of AI perspectives. I can analyze, synthesize, and even visualize complex information. Configure your AI team using the ⚙️ settings, then ask your question!"
    
    await cl.Message(content=welcome_message_content).send()

    cl.user_session.set("progress_msg", None)

# Placeholder for direct LLM invocation logic
async def invoke_direct_llm(query: str):
    print(f"invoke_direct_llm called with query: {query}")
    
    messages = [
        SystemMessage(content=DIRECT_SYSPROMPT),
        HumanMessage(content=query)
    ]
    
    response_message = cl.Message(content="")
    await response_message.send()

    async for chunk in llm_direct.astream(messages):
        if chunk.content:
            await response_message.stream_token(chunk.content)
    
    await response_message.update() # Finalize the streamed message
    return "Direct response streamed" # Test expects a return, actual content is streamed

# Placeholder for LangGraph invocation logic
async def invoke_langgraph(query: str, initial_state: InsightFlowState):
    print(f"invoke_langgraph called with query: {query}")
    
    cl.user_session.set("completed_steps_log", []) 

    progress_msg = cl.Message(content="") 
    await progress_msg.send()
    # Initial message only shows current activity, no completed steps yet.
    progress_msg.content = "**Current Activity:**\n⏳ Initializing InsightFlow process... (0%)" 
    await progress_msg.update() 
    cl.user_session.set("progress_msg", progress_msg)

    # Setup callback handler - still useful for LLM calls or other low-level events if desired
    callback_handler = InsightFlowCallbackHandler(progress_message=progress_msg)

    current_state = initial_state.copy() # Work with a copy
    current_state["query"] = query
    current_state["current_step_name"] = "planner_agent" # Reset step for new invocation

    # Check for Quick Mode and adjust personas if needed
    quick_mode_active = cl.user_session.get("quick_mode", False) # Default to False if not set
    if quick_mode_active:
        print("Quick Mode is ON. Using predefined quick mode personas.")
        current_state["selected_personas"] = list(QUICK_MODE_PERSONAS) # Ensure it's a new list copy
    # If quick_mode is OFF, selected_personas from initial_state (set by on_chat_start or commands) will be used.

    # Prepare config for LangGraph invocation (e.g., for session/thread ID)
    # In a Chainlit context, cl.user_session.get("id") can give a thread_id
    thread_id = cl.user_session.get("id", "default_thread_id") # Get Chainlit thread_id or a default
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [callback_handler] # Add our callback handler
    }

    # progress_msg = cl.Message(content="⏳ Processing with InsightFlow (0%)...") # OLD TODO
    # await progress_msg.send()
    # cl.user_session.set("progress_msg", progress_msg)

    final_state = await insight_flow_graph.ainvoke(current_state, config=config)
    
    # Final progress update
    if progress_msg:
        await progress_msg.update() # Ensure final content (100%) is sent and displayed

    # The present_results node should handle sending messages. 
    # invoke_langgraph will return the final state which on_message saves.
    return final_state

@cl.on_message
async def on_message(message: cl.Message):
    """Handles incoming user messages and routes based on direct_mode."""

    # Apply the latest settings from UI to the session and state
    await _apply_chat_settings_to_state()

    direct_mode = cl.user_session.get("direct_mode") # Values now reflect chat settings
    msg_content_lower = message.content.lower().strip()

    # Command handling (can be kept as backups or for power users)
    if msg_content_lower == "/direct on":
        cl.user_session.set("direct_mode", True)
        await cl.Message(content="Direct mode ENABLED.").send()
        return # Command processed, no further action
    elif msg_content_lower == "/direct off":
        cl.user_session.set("direct_mode", False)
        await cl.Message(content="Direct mode DISABLED.").send()
        return # Command processed, no further action
    
    # Command handling for /show perspectives
    elif msg_content_lower == "/show perspectives on":
        cl.user_session.set("show_perspectives", True)
        await cl.Message(content="Show perspectives ENABLED.").send()
        return
    elif msg_content_lower == "/show perspectives off":
        cl.user_session.set("show_perspectives", False)
        await cl.Message(content="Show perspectives DISABLED.").send()
        return

    # Command handling for /show visualization
    elif msg_content_lower == "/show visualization on":
        cl.user_session.set("show_visualization", True)
        await cl.Message(content="Show visualization ENABLED.").send()
        return
    elif msg_content_lower == "/show visualization off":
        cl.user_session.set("show_visualization", False)
        await cl.Message(content="Show visualization DISABLED.").send()
        return

    # Command handling for /quick_mode
    elif msg_content_lower == "/quick_mode on":
        cl.user_session.set("quick_mode", True)
        await cl.Message(content="Quick mode ENABLED.").send()
        return
    elif msg_content_lower == "/quick_mode off":
        cl.user_session.set("quick_mode", False)
        await cl.Message(content="Quick mode DISABLED.").send()
        return
    elif msg_content_lower == "/help": # <-- ADD /help COMMAND HANDLER
        await send_help_message()
        return

    # If not a /direct command, proceed with existing direct_mode check for LLM calls
    if direct_mode: # This direct_mode is now sourced from chat settings via _apply_chat_settings_to_state
        await invoke_direct_llm(message.content)
    else:
        insight_flow_state = cl.user_session.get("insight_flow_state")
        if not insight_flow_state:
            # Fallback if state isn't somehow initialized
            await cl.Message(content="Error: Session state not found. Please restart the chat.").send()
            return
        
        # The selected_personas in insight_flow_state have been updated by _apply_chat_settings_to_state
        # The quick_mode check within invoke_langgraph will use cl.user_session.get("quick_mode")
        # which was also updated by _apply_chat_settings_to_state.
        updated_state = await invoke_langgraph(message.content, insight_flow_state)
        cl.user_session.set("insight_flow_state", updated_state) # Save updated state

print("app.py initialized with LLMs, on_chat_start, and on_message defined")

# --- NEW EXPORT ACTION STUBS ---
@cl.action_callback("export_markdown")
async def on_export_markdown(action: cl.Action):
    # Placeholder for Markdown export logic
    await cl.Message(content=f"Markdown export for action '{action.name}' initiated (not fully implemented).").send()

@cl.action_callback("export_pdf")
async def on_export_pdf(action: cl.Action):
    # Placeholder for PDF export logic
    await cl.Message(content=f"PDF export for action '{action.name}' initiated (not fully implemented).").send()

# --- NEW FUNCTION FOR WITTY COMPLETION REMARKS ---
async def generate_witty_completion_remark(query: str, selected_persona_ids: List[str], synthesized_snippet: Optional[str]) -> str:
    if not llm_direct: # Ensure llm_direct is initialized
        print("llm_direct not initialized, cannot generate witty remark.")
        return "Intellectual journey complete! Bravo! ✔️" # Fallback

    persona_factory: PersonaFactory = cl.user_session.get("persona_factory")
    persona_names = []
    if persona_factory:
        for pid in selected_persona_ids:
            config = persona_factory.persona_configs.get(pid)
            if config and config.get("name"):
                persona_names.append(config["name"])
            else:
                persona_names.append(pid.capitalize())
    else:
        persona_names = [pid.capitalize() for pid in selected_persona_ids]
    
    persona_names_str = ", ".join(persona_names) if persona_names else "various"
    if not synthesized_snippet: synthesized_snippet = "a fascinating topic"
    if len(synthesized_snippet) > 100: synthesized_snippet = synthesized_snippet[:97] + "..."

    prompt_template = f"""You are a charming and slightly cheeky AI host, like a talk show host wrapping up a segment. 
The user just explored the query: '{query}' 
with insights from {persona_names_str} perspectives. 
The main takeaway was about: '{synthesized_snippet}'. 
Craft a short, witty, and encouraging closing remark (1-2 sentences, max 25 words) to signify the completion. 
Example: 'And that, folks, is how you dissect a universe! Until next time, keep those neurons firing!'
Another Example: 'Well, that was a delightful dive into the rabbit hole of {query}! Stay curious!'
Your remark:"""

    messages = [SystemMessage(content=prompt_template)]
    try:
        response = await llm_direct.ainvoke(messages)
        remark = response.content.strip()
        # Basic filter for overly long or nonsensical remarks if needed, though prompt should guide it
        if len(remark) > 150 or len(remark) < 10: # Arbitrary length check
            return "And that's a wrap on that fascinating exploration! What's next? ✔️" 
        return f"{remark} ✔️"
    except Exception as e:
        print(f"Error generating witty remark: {e}")
        return "Exploration complete! Well done! ✔️" # Fallback on error

# --- HELP MESSAGE FUNCTION ---
async def send_help_message():
    """Sends the detailed help message to the user."""
    help_text_md = """# Welcome to InsightFlow AI - Your Guide!

InsightFlow AI is designed to help you explore complex topics with depth and clarity by providing multiple AI-driven perspectives, synthesizing them into a coherent understanding, and even offering visual summaries. Think of it as your personal team of AI research assistants!

## What Can InsightFlow AI Do?

*   **Multi-Perspective Analysis:** Instead of a single answer, InsightFlow AI engages a team of specialized AI "personas" (e.g., Analytical, Philosophical, Scientific) to examine your query from different angles. This provides a richer, more nuanced understanding.
*   **Insight Synthesis:** The individual perspectives are then intelligently combined into a single, comprehensive synthesized response.
*   **Visualizations (Optional):**
    *   **DALL-E Sketches:** Get a conceptual, hand-drawn style visual note summarizing the key ideas.
    *   **Mermaid Diagrams:** See a concept map illustrating the relationships between your query, the personas, and the synthesized view.
*   **RAG (Retrieval Augmented Generation - for supported personas):** For certain personas, the AI can search a dedicated knowledge base of relevant documents to ground its responses in specific information, enhancing accuracy and depth.
*   **Flexible Interaction Modes:**
    *   **Full Multi-Persona Mode:** The default mode, engaging your selected team for in-depth analysis.
    *   **Direct Mode:** Get a quick, straightforward answer from a single LLM, bypassing the multi-persona/LangGraph system.
    *   **Quick Mode:** A faster multi-persona analysis using a reduced set of (currently 2) predefined personas.
*   **Exportable Results:** You'll be able to export your analysis (though this feature is still under full development).

## Why Does It Work This Way? (The Philosophy)

InsightFlow AI is built on the idea that complex topics are best understood by examining them from multiple viewpoints. Just like a team of human experts can provide more comprehensive insight than a single individual, our AI personas work together to give you a more well-rounded and deeply considered response. The structured workflow (managed by LangGraph) ensures a methodical approach to generating and synthesizing these perspectives.

## Using the Settings (⚙️ Gear Icon)

You can customize your InsightFlow AI experience using the settings panel:

*   **⚙️ Enable RAG Features:**
    *   **Functionality:** (Currently being refined) When enabled, personas designated as "RAG-enabled" (like Analytical, Philosophical, Metaphorical on the Balanced Team) will attempt to search their specific knowledge bases for relevant context before generating their perspective. This can lead to more informed and detailed answers.
    *   **Status:** Basic RAG data loading and vector store creation for personas is active. The quality and breadth of data sources are still being expanded.

*   **🎯 Persona Team:**
    *   **Functionality:** Allows you to quickly select a pre-configured team of personas. Selecting a team will override any individual persona toggles below.
    *   **Current Teams:**
        *   `🎨 Creative Synthesis Team`: (Metaphorical, Futuristic, Philosophical) - Generates novel ideas and artistic interpretations.
        *   `📊 Data-Driven Analysis Squad`: (Analytical, Factual, Scientific) - Focuses on factual accuracy and logical deduction.
        *   `⚖️ Balanced Overview Group`: (Analytical, Philosophical, Metaphorical) - **This is the default team on startup.** Provides a well-rounded perspective.
    *   **Status:** Team selection is functional.

*   **Individual Persona Toggles (e.g., Analytical, Scientific, etc.):**
    *   **Functionality:** If no team is selected (or "-- Select a Team (Optional) --" is chosen), you can manually toggle individual personas ON or OFF to form a custom team for the analysis.
    *   **Status:** Functional.

*   **🚀 Direct Mode:**
    *   **Functionality:** If ON, your query will be answered by a single, general-purpose LLM for a quick, direct response, bypassing the multi-persona/LangGraph system.
    *   **Status:** Functional.

*   **⚡ Quick Mode:**
    *   **Functionality:** If ON, InsightFlow uses a smaller, predefined set of personas (currently Analytical and Factual) for a faster multi-perspective analysis. This overrides team/individual selections when active.
    *   **Status:** Functional.

*   **👁️ Show Individual Perspectives:**
    *   **Functionality:** If ON (default), after the main synthesized response, the individual contributions from each engaged persona will also be displayed.
    *   **Status:** Functional.

*   **🎨 Show Visualizations:**
    *   **Functionality:** If ON (default), InsightFlow will attempt to generate and display a DALL-E image sketch and a Mermaid concept map related to the synthesized response.
    *   **Status:** Functional (DALL-E requires API key setup).

## Getting Started

1.  **Review Settings (⚙️):** Select a Persona Team or toggle individual personas. Ensure RAG is enabled if you want to try it with supported personas.
2.  **Ask Your Question:** Type your query into the chat and press Enter.
3.  **Observe:** Watch the progress updates as InsightFlow AI engages the personas, synthesizes insights, and generates visualizations.

We hope you find InsightFlow AI insightful!
"""
    await cl.Message(content=help_text_md).send()

async def _initialize_embedding_model_in_session(force_openai: bool = False):
    """Helper to initialize/re-initialize embedding model based on settings."""
    use_finetuned = cl.user_session.get("use_finetuned_embedding", False)
    
    # Override if force_openai is true (e.g. for initial default)
    if force_openai:
        use_finetuned = False
        cl.user_session.set("use_finetuned_embedding", False) # Ensure session reflects this override

    current_model_details = cl.user_session.get("embedding_model_details", {})
    new_model_id = ""
    new_model_type = ""

    if use_finetuned:
        new_model_id = FINETUNED_BALANCED_TEAM_EMBED_ID
        new_model_type = "hf"
    else:
        new_model_id = OPENAI_EMBED_MODEL_ID
        new_model_type = "openai"

    if current_model_details.get("id") == new_model_id and current_model_details.get("type") == new_model_type and cl.user_session.get("embedding_model_instance") is not None:
        print(f"Embedding model '{new_model_id}' ({new_model_type}) already initialized and matches settings.")
        return

    print(f"Initializing embedding model: '{new_model_id}' ({new_model_type})...")
    try:
        embedding_instance = get_embedding_model_instance(new_model_id, new_model_type)
        cl.user_session.set("embedding_model_instance", embedding_instance)
        cl.user_session.set("embedding_model_details", {"id": new_model_id, "type": new_model_type})
        print(f"Embedding model '{new_model_id}' ({new_model_type}) initialized and set in session.")
    except Exception as e:
        print(f"Error initializing embedding model '{new_model_id}': {e}")
        cl.user_session.set("embedding_model_instance", None)
        cl.user_session.set("embedding_model_details", {})