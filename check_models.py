import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from your .env file
load_dotenv()

# Configure the client with your API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in your .env file.")
else:
    genai.configure(api_key=api_key)

    print("--- Fetching Available Gemini Models ---\n")
    try:
        # List all models and check their supported methods
        for model in genai.list_models():
            # We only care about models that can generate text content
            if 'generateContent' in model.supported_generation_methods:
                print(f"Model Name: {model.name}")
                print(f"  - Description: {model.description}\n")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nThis likely means the required API is not enabled for your project.")
        print("Please go to https://console.cloud.google.com/apis/library and enable the 'Generative Language API' or 'Vertex AI API'.")