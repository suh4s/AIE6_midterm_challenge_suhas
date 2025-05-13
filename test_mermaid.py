import chainlit as cl

@cl.on_chat_start
async def start():
    # Send a simple mermaid diagram
    await cl.Message(content="# Testing Mermaid Diagram").send()
    
    # Basic diagram with minimal syntax
    mermaid_content = """```mermaid
flowchart LR
    A[Start] --> B[Process]
    B --> C[End]
    style A fill:#f9f
    style B fill:#bbf
    style C fill:#bfb
```"""

    await cl.Message(content=mermaid_content).send()

@cl.on_message
async def on_message(message: cl.Message):
    # Echo the message back with a mermaid diagram
    response = f"You said: {message.content}\n\n"
    
    # Create a dynamic diagram based on user input
    diagram = f"""```mermaid
flowchart TD
    User["{message.content}"] --> Process
    Process --> Response
    style User fill:#f9f
    style Process fill:#bbf
    style Response fill:#bfb
```"""
    
    await cl.Message(content=response + diagram).send() 