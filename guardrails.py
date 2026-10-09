"""
AI Call Assistant - Phase 4

Basic deterministic safety guardrails.

These checks happen BEFORE the LLM response.
"""

import re


# ============================================================
# Sensitive information patterns
# ============================================================

SENSITIVE_PATTERNS = [

    r"\botp\b",

    r"\bone[- ]time password\b",

    r"\bpin\b",

    r"\bpassword\b",

    r"\bpasscode\b",

    r"\bcvv\b",

    r"\bcvc\b",

    r"\bverification code\b",

    r"\bauthentication code\b",

    r"\bsecurity code\b",

    r"\bbank account number\b",

    r"\bcard number\b",

    r"\bdebit card\b",

    r"\bcredit card\b",

    r"\bnet banking password\b",

]


# ============================================================
# Check sensitive information
# ============================================================

def contains_sensitive_request(
    message: str
) -> bool:

    text = message.lower()

    for pattern in SENSITIVE_PATTERNS:

        if re.search(pattern, text):

            return True

    return False


# ============================================================
# Public guardrail function
# ============================================================

def check_guardrails(
    message: str
) -> str:

    """
    Returns one of:

    ALLOW
    BLOCK_SENSITIVE
    """

    if contains_sensitive_request(message):

        return "BLOCK_SENSITIVE"

    return "ALLOW"


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":

    tests = [

        "What is the processing fee?",

        "Can you tell me the interest rate?",

        "Please give me your OTP.",

        "Tell me your PIN.",

        "What is the loan tenure?",

        "Give me your password."

    ]

    print()
    print("=" * 60)
    print("        GUARDRAIL TEST")
    print("=" * 60)

    for message in tests:

        result = check_guardrails(message)

        print()
        print("Message:", message)
        print("Result :", result)