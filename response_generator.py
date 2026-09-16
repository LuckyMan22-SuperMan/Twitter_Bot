from retrieval import retrieve_similar_cases
from intent_classifier import predict_intent

import os
from dotenv import load_dotenv
from google import genai


# ============================================
# 1. LOAD API KEY
# ============================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found.")

client = genai.Client(api_key=api_key)


# ============================================
# 2. RESPONSE GENERATION
# ============================================

def generate_response(
    customer_message,
    predicted_intent,
    retrieved_cases
):
    """
    Generate a customer-support response using
    historical AppleSupport responses as evidence.
    """

    # ----------------------------------------
    # Check for historical evidence
    # ----------------------------------------

    if retrieved_cases.empty:
        return {
            "response": "",
            "status": "INSUFFICIENT_EVIDENCE"
        }

    # ----------------------------------------
    # Build historical evidence
    # ----------------------------------------

    evidence = ""

    for i, (_, row) in enumerate(
        retrieved_cases.iterrows(),
        start=1
    ):
        evidence += f"""
Case {i}:
Customer:
{row["clean_customer_text"]}

AppleSupport:
{row["brand_response"]}

Similarity:
{row["similarity"]:.4f}

"""

    # ----------------------------------------
    # Build grounded prompt
    # ----------------------------------------

    prompt = f"""
You are an AI customer-support assistant.

Your task is to draft a concise response to a
customer using ONLY the historical AppleSupport
responses provided below as evidence.

Customer message:
{customer_message}

Predicted intent:
{predicted_intent}

Historical AppleSupport evidence:
{evidence}

Instructions:

1. Use the historical AppleSupport responses as
   your primary source of evidence.

2. Do not invent troubleshooting steps, product
   facts, policies, or solutions that are not
   supported by the evidence.

3. Do not claim that AppleSupport previously said
   something unless it is actually present in
   the evidence.

4. You may combine compatible information from
   multiple historical responses, but only when
   the information is directly relevant to the
   customer's current problem.

5. Prioritize the most relevant evidence based on
   the customer's specific message and predicted
   intent. Do not include information simply
   because it appears in one of the retrieved cases.

6. If a historical response asks for information
   that is not relevant to the current customer's
   problem, do not include that question.

7. If the evidence is insufficient to provide a
   useful and grounded response, return exactly:
   INSUFFICIENT_EVIDENCE

8. Keep the response concise and professional,
   like a real customer-support message.

9. If the historical responses mainly ask the
   customer for information, it is acceptable to
   ask for similar information rather than invent
   a technical solution.

Return only the customer-support response or
INSUFFICIENT_EVIDENCE.
"""

    # ----------------------------------------
    # Call Gemini
    # ----------------------------------------

    print("Prompt length:", len(prompt), "characters")
    print("Sending request to Gemini...")

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    generated_text = interaction.output_text.strip()

    # ----------------------------------------
    # Handle insufficient evidence
    # ----------------------------------------

    if generated_text == "INSUFFICIENT_EVIDENCE":
        return {
            "response": "",
            "status": "INSUFFICIENT_EVIDENCE"
        }

    return {
        "response": generated_text,
        "status": "GENERATED"
    }


# ============================================
# 3. END-TO-END TEST
# ============================================

if __name__ == "__main__":

    # ----------------------------------------
    # Customer message
    # ----------------------------------------

    customer_message = (
        "My iPhone battery is draining really fast"
    )

    # ----------------------------------------
    # Predict intent
    # ----------------------------------------

    predicted_intent, confidence = predict_intent(
        customer_message
    )

    print("\nPredicted intent:")
    print(predicted_intent)

    print("Intent confidence:")
    print(round(confidence, 4))

    # ----------------------------------------
    # Retrieve historical cases
    # ----------------------------------------

    retrieved_cases = retrieve_similar_cases(
        customer_message,
        predicted_intent,
        top_k=3
    )

    print("\nRetrieved cases:", len(retrieved_cases))

    # ----------------------------------------
    # Generate response
    # ----------------------------------------

    result = generate_response(
        customer_message,
        predicted_intent,
        retrieved_cases
    )

    print("\nGenerated response:")
    print(result["response"])

    print("\nStatus:")
    print(result["status"])