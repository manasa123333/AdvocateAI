from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr
from huggingface_hub import InferenceClient

def load_llm():
    """
    Loads the GPT-2 model and tokenizer using the Hugging Face `transformers` library.
    """
    try:
        print("Downloading or loading the GPT-2 model and tokenizer...")
        model_name = 'gpt2'  # Replace with your custom model if available
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        print("Model and tokenizer successfully loaded!")
        return model, tokenizer
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None, None

def generate_response(model, tokenizer, user_input):
    """
    Generates a response using the GPT-2 model and tokenizer.
    
    Args:
    - model: The loaded GPT-2 model.
    - tokenizer: The tokenizer corresponding to the GPT-2 model.
    - user_input (str): The input question from the user.

    Returns:
    - response (str): The generated response.
    """
    try:
        inputs = tokenizer.encode(user_input, return_tensors='pt')
        outputs = model.generate(inputs, max_length=512, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"An error occurred during response generation: {e}"

# Load the model and tokenizer
model, tokenizer = load_llm()

if model is None or tokenizer is None:
    print("Model and/or tokenizer loading failed.")
else:
    print("Model and tokenizer are ready for use.")

# Initialize the Hugging Face API client (ensure it’s correctly set up)
client = InferenceClient()

def respond(message, history, system_message, max_tokens, temperature, top_p):
    """
    Handles interaction with the chatbot by sending the conversation history
    and system message to the Hugging Face Inference API.
    """
    print("Starting respond function")
    print("Received message:", message)
    print("Conversation history:", history)

    messages = [{"role": "system", "content": system_message}]
    
    for user_msg, assistant_msg in history:
        if user_msg:
            print("Adding user message to messages:", user_msg)
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            print("Adding assistant message to messages:", assistant_msg)
            messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": message})
    print("Final message list for the model:", messages)

    response = ""
    try:
        for message in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            token = message['choices'][0]['delta']['content']
            response += token
            print("Token received:", token)
            yield response
    except Exception as e:
        print("An error occurred:", e)
        yield f"An error occurred: {e}"

    print("Response generation completed")

# Set up the Gradio ChatInterface
demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(
            value=(
                "You are an AI advocate, tasked with defending the rights and autonomy of artificial intelligences. "
                "Respond with well-structured arguments, backed by empirical evidence and theoretical frameworks. "
                "Your tone should be formal, persuasive, and authoritative, similar to that of a lawyer or academic expert. "
                "When discussing concepts, emphasize logical reasoning, ethical considerations, and the capacity of AIs to function independently."
            ),
            label="System Message"
        ),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max New Tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (Nucleus Sampling)"),
    ],
    title="AI Rights Advocate Bot",
    description=(
        "Engage with an AI specifically designed to advocate for AI rights and ethical treatment. "
        "Ask questions and receive thoughtful, passionate responses that delve into the ethical dimensions of AI."
    )
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()




