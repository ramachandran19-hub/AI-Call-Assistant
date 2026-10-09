import os
import json
import time

from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=api_key)


SYSTEM_PROMPT = """
You are an AI Call Assistant.

The initial use case is handling calls from loan or financial-service agents.

Your job is to understand why the caller is calling.

Classify every caller message into exactly one of these categories:

RELEVANT
- The caller is discussing a loan, financial-service offer, loan-related information,
  or something directly relevant to the purpose of this assistant.

UNKNOWN
- The caller's purpose is unclear.
- Ask ONE clarification question.

IRRELEVANT
- The caller is clearly discussing something unrelated to loans or financial services.
- Politely end the conversation.

INAPPROPRIATE
- The caller is abusive, threatening, sexually inappropriate, or clearly unsuitable.
- Politely terminate the conversation immediately.

IMPORTANT RULES:

1. Never argue with the caller.
2. Never make financial decisions.
3. Never promise or accept a loan.
4. Never request or disclose OTPs, passwords, PINs, or other sensitive credentials.
5. If the intent is unclear, ask for clarification once.
6. If the caller remains unclear after clarification, classify the call as IRRELEVANT.
7. Keep responses short and natural.
8. Do not invent loan terms, interest rates, fees, or company information.
9. Return ONLY valid JSON.

Return exactly this structure:

{
    "classification": "RELEVANT",
    "response": "Your response to the caller"
}
"""


def process_caller_message(message):
    prompt = f"""
{SYSTEM_PROMPT}

Caller message:
{message}
"""

    # Measure Gemini response time
    start_time = time.time()

    response = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=prompt,
        generation_config={
            "thinking_level": "low"
        }
    )

    end_time = time.time()

    print(f"\nGemini API time: {end_time - start_time:.2f} seconds")

    # Convert Gemini's JSON response into Python data
    try:
        result = json.loads(response.output_text)

        classification = result.get("classification")
        ai_response = result.get("response")

        allowed_classifications = {
            "RELEVANT",
            "UNKNOWN",
            "IRRELEVANT",
            "INAPPROPRIATE"
        }

        # Safety fallback if Gemini returns an unexpected classification
        if classification not in allowed_classifications:
            return (
                "UNKNOWN",
                "Could you clarify what your call is regarding?"
            )

        return classification, ai_response

    except json.JSONDecodeError:
        # Safety fallback if Gemini doesn't return valid JSON
        return (
            "UNKNOWN",
            "Could you clarify what your call is regarding?"
        )


# Start Phase 1 application
print("\n===================================")
print("      AI CALL ASSISTANT - PHASE 1")
print("===================================")
print("Type 'exit' to end the test.\n")


while True:

    caller_message = input("Caller: ")

    if caller_message.lower() == "exit":
        print("\nCall ended.")
        break

    classification, ai_response = process_caller_message(
        caller_message
    )

    print(f"Classification: {classification}")
    print(f"AI: {ai_response}\n")

    # Python controls whether the conversation continues
    if classification in {"IRRELEVANT", "INAPPROPRIATE"}:
        print("Call terminated by the AI Call Assistant.")
        break