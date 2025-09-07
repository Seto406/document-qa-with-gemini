import os
from huggingface_hub import InferenceClient

# --- Paste your token here ---
HUGGINGFACE_API_TOKEN = "hf_YourTokenGoesHere"

print("Attempting to connect to Hugging Face...")

try:
    # Initialize the client directly with the token
    client = InferenceClient(token=HUGGINGFACE_API_TOKEN)

    # Make a simple request to a reliable model
    response = client.text_generation(
        prompt="The title of the paper is",
        model="google/flan-t5-large",
        max_new_tokens=10
    )

    print("\n--- SUCCESS! ---")
    print("Connection successful and received a response:")
    print(response)
    print("\nThis confirms your API token is working correctly.")

except Exception as e:
    print("\n--- FAILED ---")
    print("The connection failed. This confirms an issue with your token or network.")
    print("Error details:", e)