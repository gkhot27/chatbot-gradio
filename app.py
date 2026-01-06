import gradio as gr
from openai import OpenAI
import os

# Load environment variables (works for both local .env and Hugging Face Spaces secrets)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not needed on Hugging Face Spaces

# Initialize OpenAI client
# On Hugging Face Spaces, API key is set as a Secret
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your environment variables or Hugging Face Space secrets.")

client = OpenAI(api_key=api_key)

def chat_stream(message, history):
    """
    Streaming chat function for Gradio
    """
    # Convert Gradio history format to OpenAI format
    messages = [
        {"role": "system", "content": "You are a funny and sarcastic chat bot "}
    ] 
    # Add conversation history
    if history:
        for turn in history:
            try:
                if isinstance(turn, dict):
                    # New Gradio 6.0+ format: dictionary with role and content
                    role = turn.get("role", "")
                    content = turn.get("content", "")
                    if role and content:
                        # Already in OpenAI format, just add it
                        messages.append({"role": role, "content": str(content)})
                elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
                    # Legacy format: [user_msg, assistant_msg]
                    user_msg = turn[0]
                    assistant_msg = turn[1]
                    if user_msg:
                        messages.append({"role": "user", "content": str(user_msg)})
                    if assistant_msg:
                        messages.append({"role": "assistant", "content": str(assistant_msg)})
            except (IndexError, KeyError, TypeError, AttributeError):
                # Skip malformed history entries
                continue
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Stream the response
    full_response = ""
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=1000
    )
    
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        if content:
            full_response += content
            yield full_response
    
    return full_response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_stream,
    title="FUNNY CHATBOT",
    description="Chat with an AI assistant powered by OpenAI",
    examples=[
        "Tell me a joke",
        "Why is AI taking my job?",
        "What is the meaning of life?"
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    # For Hugging Face Spaces, use default launch settings
    # Theme is set via the Space's README.md metadata or can be customized in ChatInterface
    demo.launch()

