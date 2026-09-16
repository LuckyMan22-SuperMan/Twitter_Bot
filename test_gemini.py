from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

print("Sending request...")

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input="Say hello in one short sentence."
)

print("Response:")
print(interaction.output_text)