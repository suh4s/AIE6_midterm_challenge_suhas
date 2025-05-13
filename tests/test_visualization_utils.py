import pytest
from unittest.mock import AsyncMock, MagicMock
from openai import AsyncOpenAI
from openai.types.images_response import Image, ImagesResponse # For mocking response
from typing import Optional

from utils.visualization_utils import generate_dalle_image

# Placeholder for the actual function for now, so the test file can be written
# async def generate_dalle_image(prompt: str, client: AsyncOpenAI) -> Optional[str]:
#     pass

@pytest.mark.asyncio
async def test_generate_dalle_image_success():
    """Test successful DALL-E image generation."""
    mock_client = AsyncMock(spec=AsyncOpenAI)
    expected_url = "https://example.com/image.png"
    
    # Mocking the response structure from OpenAI DALL-E API
    # It returns an ImagesResponse object, which has a 'data' attribute (list of Image objects)
    # Each Image object has a 'url' attribute.
    mock_image = Image(b64_json=None, revised_prompt=None, url=expected_url)
    mock_images_response = ImagesResponse(created=1234567890, data=[mock_image])
    
    mock_client.images.generate = AsyncMock(return_value=mock_images_response)
    
    prompt_text = "A beautiful sunset over mountains, hand-drawn sketch style"
    image_url = await generate_dalle_image(prompt_text, mock_client)
    
    mock_client.images.generate.assert_called_once_with(
        prompt=prompt_text,
        model="dall-e-3",
        size="1024x1024",
        quality="standard", # DALL-E 3 only supports 'standard' and 'hd'
        n=1,
        style="vivid" # 'vivid' or 'natural'. 'vivid' for hyper-real, 'natural' for less so.
                      # The "hand-drawn" aspect is best achieved via prompt engineering.
    )
    assert image_url == expected_url

@pytest.mark.asyncio
async def test_generate_dalle_image_api_error():
    """Test DALL-E image generation when API returns an error."""
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_client.images.generate = AsyncMock(side_effect=Exception("API Error"))
    
    prompt_text = "A complex abstract concept"
    image_url = await generate_dalle_image(prompt_text, mock_client)
    
    mock_client.images.generate.assert_called_once()
    assert image_url is None

@pytest.mark.asyncio
async def test_generate_dalle_image_no_data_in_response():
    """Test DALL-E image generation when API response has no data."""
    mock_client = AsyncMock(spec=AsyncOpenAI)
    # Simulate a response where the 'data' list is empty or None
    mock_images_response_no_data = ImagesResponse(created=1234567890, data=[])
    mock_client.images.generate = AsyncMock(return_value=mock_images_response_no_data)
    
    prompt_text = "A rare scenario"
    image_url = await generate_dalle_image(prompt_text, mock_client)
    
    mock_client.images.generate.assert_called_once()
    assert image_url is None

@pytest.mark.asyncio
async def test_generate_dalle_image_no_url_in_image_object():
    """Test DALL-E image generation when Image object in response has no URL."""
    mock_client = AsyncMock(spec=AsyncOpenAI)
    # Simulate a response where the Image object has no 'url'
    mock_image_no_url = Image(b64_json=None, revised_prompt=None, url=None) # type: ignore
    mock_images_response_no_url = ImagesResponse(created=1234567890, data=[mock_image_no_url])

    mock_client.images.generate = AsyncMock(return_value=mock_images_response_no_url)
    
    prompt_text = "Another rare scenario"
    image_url = await generate_dalle_image(prompt_text, mock_client)
    
    mock_client.images.generate.assert_called_once()
    assert image_url is None

# --- Tests for generate_mermaid_code --- #
from langchain_openai import ChatOpenAI # For type hinting the mock LLM client
from langchain_core.messages import AIMessage # For mocking LLM response

# Placeholder for the function to be tested
# from utils.visualization_utils import generate_mermaid_code 

MERMAID_SYSTEM_PROMPT_TEMPLATE = """You are an expert in creating Mermaid diagrams. Based on the following text, generate a concise and accurate Mermaid diagram syntax. Only output the Mermaid code block (```mermaid\n...
```). Do not include any other explanatory text. If the text cannot be reasonably converted to a diagram, output '// No suitable diagram' as a comment. Text: {text_input}
"""

@pytest.mark.asyncio
async def test_generate_mermaid_code_success():
    """Test successful Mermaid code generation."""
    mock_llm_client = AsyncMock(spec=ChatOpenAI)
    expected_mermaid_code = "graph TD;\n    A[Start] --> B{Is it?};\n    B -- Yes --> C[End];\n    B -- No --> D[Alternative End];"
    
    # Mock the response from the LLM client's ainvoke method
    mock_llm_response = AIMessage(content=f"```mermaid\n{expected_mermaid_code}\n```")
    mock_llm_client.ainvoke = AsyncMock(return_value=mock_llm_response)
    
    input_text = "Describe a simple decision process."
    
    # Dynamically import generate_mermaid_code to ensure mocks are applied if it uses module-level things
    from utils.visualization_utils import generate_mermaid_code
    mermaid_output = await generate_mermaid_code(input_text, mock_llm_client)
    
    expected_prompt = MERMAID_SYSTEM_PROMPT_TEMPLATE.format(text_input=input_text)
    # ainvoke is called with a list of messages or a string. Let's assume a list with a SystemMessage for now.
    # We will need to check the actual implementation of generate_mermaid_code to refine this assertion.
    # For now, let's assume it sends a list of messages, and the first one contains the prompt.
    # Or, if it sends a string directly to ainvoke, we adapt.
    # Based on typical Langchain usage with ChatModels, it's usually a list of Messages.
    
    # Check that ainvoke was called. For the prompt content, we'll check the first message's content.
    mock_llm_client.ainvoke.assert_called_once()
    called_messages = mock_llm_client.ainvoke.call_args[0][0]
    assert len(called_messages) > 0
    assert called_messages[0].content == expected_prompt # Assuming SystemMessage is the first
    
    assert mermaid_output == expected_mermaid_code

@pytest.mark.asyncio
async def test_generate_mermaid_code_llm_error():
    """Test Mermaid code generation when LLM call fails."""
    mock_llm_client = AsyncMock(spec=ChatOpenAI)
    mock_llm_client.ainvoke = AsyncMock(side_effect=Exception("LLM API Error"))
    
    input_text = "Some complex text that might cause an error."
    from utils.visualization_utils import generate_mermaid_code
    mermaid_output = await generate_mermaid_code(input_text, mock_llm_client)
    
    mock_llm_client.ainvoke.assert_called_once()
    assert mermaid_output is None # Expect None or a specific error string

@pytest.mark.asyncio
async def test_generate_mermaid_code_no_suitable_diagram():
    """Test Mermaid code generation when LLM indicates no suitable diagram."""
    mock_llm_client = AsyncMock(spec=ChatOpenAI)
    # LLM returns the specific comment indicating no diagram
    mock_llm_response = AIMessage(content="// No suitable diagram")
    mock_llm_client.ainvoke = AsyncMock(return_value=mock_llm_response)
    
    input_text = "This text is not suitable for a diagram."
    from utils.visualization_utils import generate_mermaid_code
    mermaid_output = await generate_mermaid_code(input_text, mock_llm_client)
    
    mock_llm_client.ainvoke.assert_called_once()
    # Depending on implementation, it might return None or the comment itself.
    # For now, let's assume it should return None if it sees this specific comment.
    assert mermaid_output is None 