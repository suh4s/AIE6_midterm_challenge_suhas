# Placeholder for visualization utilities 

from openai import AsyncOpenAI
from openai.types.images_response import Image, ImagesResponse
from typing import Optional
import asyncio # For potential timeout, though not strictly required by tests yet
import httpx # Added for timeout configuration in generate_dalle_image
from langchain_openai import ChatOpenAI # For type hinting generate_mermaid_code
from langchain_core.messages import SystemMessage # For generate_mermaid_code
import re # For extracting mermaid code

# System prompt for Mermaid generation (copied from test file for consistency)
MERMAID_SYSTEM_PROMPT_TEMPLATE = """You are an expert in creating Mermaid diagrams. Based on the following text, generate a concise and accurate Mermaid diagram syntax. Only output the Mermaid code block (```mermaid\n...
```). Do not include any other explanatory text. If the text cannot be reasonably converted to a diagram, output '// No suitable diagram' as a comment. Text: {text_input}
"""

async def generate_dalle_image(prompt: str, client: AsyncOpenAI) -> Optional[str]:
    """Generates an image using DALL-E 3 and returns the URL."""
    try:
        print(f"Generating DALL-E image for prompt: '{prompt[:100]}...'")
        # Configure a timeout for the API call
        # Note: httpx.AsyncClient allows request-specific timeouts.
        # AsyncOpenAI uses httpx internally. Default timeout for AsyncOpenAI is 60s.
        # We can either rely on its default or, if we needed finer control over this specific call,
        # we might need to configure the client upon its instantiation or use a separate client.
        # For now, let's assume the default client timeout is sufficient, or add a note about it.
        # Default timeout is 1 minute. For DALL-E, this is usually enough.
        # For more control, you can pass `timeout=httpx.Timeout(30.0, connect=5.0)` to AsyncOpenAI client init.

        response = await client.images.generate(
            prompt=prompt,
            model="dall-e-3",
            size="1024x1024",
            quality="standard", # 'standard' or 'hd'. 'hd' is more detailed but might be slower/costlier.
            n=1,
            style="vivid" # 'vivid' (hyper-real and dramatic) or 'natural' (more natural, less hyper-real)
        )
        if response.data and response.data[0].url:
            return response.data[0].url
        else:
            print("DALL-E API call succeeded but returned no data or URL.")
            return None
    except Exception as e:
        print(f"An error occurred during DALL-E image generation: {e}")
        return None

async def generate_mermaid_code(text_input: str, llm_client: ChatOpenAI) -> Optional[str]:
    """Generates Mermaid diagram code from text using an LLM."""
    if not text_input or not llm_client:
        return None

    prompt = MERMAID_SYSTEM_PROMPT_TEMPLATE.format(text_input=text_input)
    messages = [SystemMessage(content=prompt)]

    try:
        print(f"Generating Mermaid code for text: '{text_input[:100]}...'")
        response = await llm_client.ainvoke(messages)
        content = response.content

        if "// No suitable diagram" in content:
            print("LLM indicated no suitable diagram for Mermaid generation.")
            return None

        # Extract content within ```mermaid ... ``` block
        match = re.search(r"```mermaid\n(.*?)\n```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            # If no block found, but also no "// No suitable diagram" comment,
            # it might be that the LLM failed to follow instructions or returned plain text.
            # We could return the raw content, or None, or try to sanitize.
            # For now, if it's not a proper block and not the explicit no-diagram comment, assume failure to follow format.
            print(f"Mermaid LLM did not return a valid Mermaid block. Raw output: {content[:200]}...")
            return None 
            
    except Exception as e:
        print(f"An error occurred during Mermaid code generation: {e}")
        return None 