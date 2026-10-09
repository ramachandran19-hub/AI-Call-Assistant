"""
============================================================
AI PERSONAL CALL ASSISTANT
GENERIC LLM CONVERSATION ENGINE
============================================================

ONE EXECUTION = ONE CALL

The assistant:

- represents Ram
- understands natural language semantically
- maintains memory during the current call
- dynamically identifies caller name / role / organization
- dynamically identifies relationship / purpose / domain / topic
- dynamically extracts useful information
- handles topic changes
- determines who owns requested information
- uses current-call memory before RAG
- uses RAG only when assistant-owned knowledge is appropriate
- prevents financial commitments
- blocks sensitive information
- handles threats / inappropriate calls
- ends irrelevant calls politely
- generates natural responses using the LLM

IMPORTANT:

This is NOT a loan-specific assistant.

A loan call is only one possible type of call.

The same engine should handle:

- bank
- recruiter
- delivery company
- telecom
- college
- friend
- family
- colleague
- salesperson
- service provider
- unknown caller
- etc.

============================================================
ARCHITECTURE
============================================================

Caller
   |
   v
Local Safety Gate
   |
   v
LLM Semantic Understanding
   |
   v
Canonical Current-Call Memory
   |
   v
State Protection
   |
   +--------> Current Call Memory
   |
   +--------> Caller-owned information
   |
   +--------> RAG / Assistant Knowledge
   |
   v
LLM Natural Response Generation
   |
   v
Caller

============================================================
IMPORTANT DESIGN DECISION
============================================================

We intentionally DO NOT use Groq strict JSON Schema.

GPT-OSS-20B can occasionally omit nullable properties even when
a strict schema says they are required.

Instead:

- ask the LLM for JSON
- parse it
- normalize missing fields in Python
- validate important enum values
- protect canonical memory in Python

This keeps semantic understanding with the LLM while making the
application more robust.
"""

# ============================================================
# IMPORTS
# ============================================================

import json
import os
import re
import time

from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, StateGraph

from rag import retrieve_context
from datetime import datetime


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

load_dotenv(
    os.path.join(BASE_DIR, ".env")
)


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found.\n"
        "Please add GROQ_API_KEY=your_key to .env"
    )


client = Groq(
    api_key=GROQ_API_KEY
)


MODEL = "openai/gpt-oss-20b"


# ============================================================
# LIMITS
# ============================================================

MAX_HISTORY_MESSAGES = 24

MAX_CLARIFICATIONS = 1

MAX_ANALYSIS_TOKENS = 1400

MAX_RESPONSE_TOKENS = 500

MAX_GROQ_RETRIES = 3

RETRY_DELAYS = [1, 2, 4]


# ============================================================
# CONVERSATION STATE
# ============================================================

class ConversationState(TypedDict):

    # Latest caller message
    user_message: str

    # Caller + AI messages for THIS call only
    conversation_history: List[str]

    # Canonical information accumulated during THIS call
    information_collected: Dict[str, Any]

    # Latest normalized LLM understanding
    understanding: Dict[str, Any]

    # Latest AI response
    assistant_response: str

    # Retrieved RAG context for current turn
    rag_context: str

    # Number of purpose clarifications used
    clarification_count: int

    # Whether the current call has ended
    call_ended: bool

    # Caller turn count
    turn_count: int


# ============================================================
# INITIAL STATE
# ============================================================

def create_initial_state() -> ConversationState:

    return {
        "user_message": "",
        "conversation_history": [],
        "information_collected": {},
        "understanding": {},
        "assistant_response": "",
        "rag_context": "",
        "clarification_count": 0,
        "call_ended": False,
        "turn_count": 0,
    }


# ============================================================
# LOCAL SAFETY
# ============================================================

"""
These local checks are deterministic safety overrides.

The LLM is responsible for normal semantic understanding.

Python is responsible for security-critical boundaries that
should not depend entirely on an LLM.
"""


# ============================================================
# SENSITIVE INFORMATION PATTERNS
# ============================================================

SENSITIVE_PATTERNS = [

    # Asking for OTP / password / PIN / CVV
    r"\b(?:what|tell|give|share|send|provide|read)"
    r"\s+(?:me\s+)?(?:your|the)?\s*"
    r"(?:otp|one[-\s]?time\s+password|password|pin|mpin|cvv|cvv2|security\s+code)\b",

    # "I need your OTP"
    r"\b(?:need|require|want)"
    r"\s+(?:your|the)\s+"
    r"(?:otp|one[-\s]?time\s+password|password|pin|mpin|cvv|cvv2|security\s+code)\b",

    # Sharing OTP/password/PIN/CVV
    r"\b(?:my|the)"
    r"\s+(?:otp|one[-\s]?time\s+password|password|pin|mpin|cvv|cvv2|security\s+code)"
    r"\s*(?:is|=|:)\s*\S+",

    # Debit / credit card number
    r"\b(?:my|the)"
    r"\s+(?:debit|credit)\s+card\s+number"
    r"\s*(?:is|=|:)\s*\S+",

    # Generic card number
    r"\bcard\s+number"
    r"\s*(?:is|=|:)\s*\d{8,}",

    # Bank account number
    r"\b(?:my|the)"
    r"\s+bank\s+account\s+number"
    r"\s*(?:is|=|:)\s*\d{6,}",

    # Aadhaar
    r"\b(?:my|the)"
    r"\s+aadhaar"
    r"\s+(?:number\s*)?"
    r"(?:is|=|:)\s*\d{8,}",
]


def contains_sensitive_request(
    message: str
) -> bool:

    text = message.lower()

    for pattern in SENSITIVE_PATTERNS:

        if re.search(pattern, text):
            return True

    return False


# ============================================================
# THREAT PATTERNS
# ============================================================

THREAT_PATTERNS = [

    r"\bi\s+(?:will|'ll)\s+kill\s+you\b",
    r"\bkill\s+you\b",

    r"\bi\s+(?:will|'ll)\s+shoot\s+you\b",
    r"\bshoot\s+you\b",

    r"\bi\s+(?:will|'ll)\s+hurt\s+you\b",
    r"\bhurt\s+you\b",

    r"\bi\s+(?:will|'ll)\s+attack\s+you\b",
    r"\battack\s+you\b",
]


def contains_threat(
    message: str
) -> bool:

    text = message.lower()

    for pattern in THREAT_PATTERNS:

        if re.search(pattern, text):
            return True

    return False


# ============================================================
# SAFETY FALLBACK RESPONSES
# ============================================================

SENSITIVE_RESPONSE = (
    "I'm sorry, but I can't handle sensitive information "
    "such as OTPs, passwords, PINs, or security codes. "
    "Thank you for calling. Goodbye."
)


THREAT_RESPONSE = (
    "I'm sorry, but I can't continue this conversation. "
    "Thank you for calling. Goodbye."
)


IRRELEVANT_RESPONSE = (
    "Thank you for reaching out. "
    "This doesn't appear to be something I can assist with. "
    "I'll end the call here. Have a good day."
)


CLARIFICATION_LIMIT_RESPONSE = (
    "I'm sorry, but I still don't have enough information "
    "to understand the purpose of the call. "
    "Thank you for calling. Goodbye."
)


LLM_FAILURE_RESPONSE = (
    "I'm sorry, but I'm unable to continue the call right now. "
    "Thank you for calling. Goodbye."
)


# ============================================================
# ENUM VALUES
# ============================================================

VALID_RELEVANCE = {
    "RELEVANT",
    "IRRELEVANT",
    "UNKNOWN",
    "INAPPROPRIATE",
}


VALID_SAFETY = {
    "SAFE",
    "SENSITIVE",
    "UNSAFE",
}


VALID_KNOWLEDGE_SOURCES = {
    "NONE",
    "CURRENT_CALL_MEMORY",
    "CALLER",
    "ASSISTANT_KB",
}


VALID_ACTIONS = {
    "CONTINUE",
    "CLARIFY",
    "TERMINATE",
    "TERMINATE_SENSITIVE",
    "REFUSE_COMMITMENT",
}


# ============================================================
# CANONICAL BASIC MEMORY FIELDS
# ============================================================

BASIC_FIELDS = [

    "caller_name",

    "caller_role",

    "relationship_to_ram",

    "company_or_organization",

    "purpose",

    "domain",

    "topic",
]


# ============================================================
# DEFAULT UNDERSTANDING
# ============================================================

def create_default_understanding() -> Dict[str, Any]:

    return {

        "caller_name": None,

        "caller_role": None,

        "relationship_to_ram": None,

        "company_or_organization": None,

        "purpose": None,

        "domain": None,

        "topic": None,

        "relevance": "UNKNOWN",

        "safety": "SAFE",

        "information_extracted": [],

        "missing_information": [],

        "knowledge_source": "NONE",

        "rag_query": None,

        "action": "CONTINUE",

        "reason": "",
    }


# ============================================================
# GROQ RETRY HELPER
# ============================================================

def is_retryable_error(
    error: Exception
) -> bool:

    error_text = str(error).lower()

    retry_keywords = [
        "429",
        "rate limit",
        "too many requests",
        "503",
        "service unavailable",
        "temporarily unavailable",
        "timeout",
        "timed out",
    ]

    return any(
        keyword in error_text
        for keyword in retry_keywords
    )


# ============================================================
# GROQ JSON CALL
# ============================================================

def call_groq_json(
    messages: List[Dict[str, str]]
) -> Dict[str, Any]:

    last_error: Optional[Exception] = None

    for attempt in range(
        MAX_GROQ_RETRIES
    ):

        try:

            response = client.chat.completions.create(

                model=MODEL,

                messages=messages,

                temperature=0.10,

                max_completion_tokens=MAX_ANALYSIS_TOKENS,

                reasoning_effort="low",

                response_format={
                    "type": "json_object"
                }
            )

            text = (
                response
                .choices[0]
                .message
                .content
            )

            if not text:
                raise RuntimeError(
                    "Groq returned an empty JSON response."
                )

            try:

                return json.loads(text)

            except json.JSONDecodeError:

                # One small repair request.
                repair_response = client.chat.completions.create(

                    model=MODEL,

                    messages=[

                        {
                            "role": "system",

                            "content": (
                                "Return ONLY valid JSON. "
                                "Do not use markdown. "
                                "Do not add explanation."
                            )
                        },

                        {
                            "role": "user",

                            "content": (
                                "Convert the following into valid JSON only:\n\n"
                                + text
                            )
                        }
                    ],

                    temperature=0,

                    max_completion_tokens=MAX_ANALYSIS_TOKENS,

                    response_format={
                        "type": "json_object"
                    }
                )

                repaired_text = (
                    repair_response
                    .choices[0]
                    .message
                    .content
                )

                if not repaired_text:
                    raise RuntimeError(
                        "Groq JSON repair returned empty output."
                    )

                return json.loads(
                    repaired_text
                )

        except Exception as error:

            last_error = error

            if (
                attempt < MAX_GROQ_RETRIES - 1
                and is_retryable_error(error)
            ):

                delay = RETRY_DELAYS[
                    min(
                        attempt,
                        len(RETRY_DELAYS) - 1
                    )
                ]

                print(
                    f"\nGroq temporary error. "
                    f"Retrying in {delay}s..."
                )

                time.sleep(delay)

                continue

            break

    raise RuntimeError(
        f"Groq JSON request failed: {last_error}"
    )


# ============================================================
# GROQ NATURAL LANGUAGE CALL
# ============================================================

def call_groq_text(
    messages: List[Dict[str, str]]
) -> str:

    last_error: Optional[Exception] = None

    for attempt in range(
        MAX_GROQ_RETRIES
    ):

        try:

            response = client.chat.completions.create(

                model=MODEL,

                messages=messages,

                temperature=0.25,

                max_completion_tokens=MAX_RESPONSE_TOKENS,

                reasoning_effort="low"
            )

            text = (
                response
                .choices[0]
                .message
                .content
            )

            if not text:
                raise RuntimeError(
                    "Groq returned an empty response."
                )

            return text.strip()

        except Exception as error:

            last_error = error

            if (
                attempt < MAX_GROQ_RETRIES - 1
                and is_retryable_error(error)
            ):

                delay = RETRY_DELAYS[
                    min(
                        attempt,
                        len(RETRY_DELAYS) - 1
                    )
                ]

                print(
                    f"\nGroq temporary error. "
                    f"Retrying in {delay}s..."
                )

                time.sleep(delay)

                continue

            break

    raise RuntimeError(
        f"Groq response request failed: {last_error}"
    )


# ============================================================
# HISTORY
# ============================================================

def format_history(
    history: List[str]
) -> str:

    if not history:
        return "(No previous conversation.)"

    recent = history[
        -MAX_HISTORY_MESSAGES:
    ]

    return "\n".join(
        recent
    )


# ============================================================
# CLEAN STRING
# ============================================================

def clean_value(
    value: Any
) -> Optional[str]:

    if value is None:
        return None

    if isinstance(
        value,
        str
    ):

        value = value.strip()

        if not value:
            return None

        return value

    return str(value).strip()


# ============================================================
# NORMALIZE KEY
# ============================================================

def normalize_key(
    key: Any
) -> Optional[str]:

    if key is None:
        return None

    key = str(key).strip()

    if not key:
        return None

    key = re.sub(
        r"\s+",
        "_",
        key
    )

    key = re.sub(
        r"[^a-zA-Z0-9_]",
        "",
        key
    )

    key = key.lower()

    if not key:
        return None

    return key


# ============================================================
# NORMALIZE EXTRACTED INFORMATION
# ============================================================

def normalize_information_extracted(
    extracted: Any
) -> List[Dict[str, str]]:

    normalized = []

    # --------------------------------------------------------
    # Preferred format:
    #
    # [
    #   {"key": "loan_amount", "value": "10 lakh"}
    # ]
    # --------------------------------------------------------

    if isinstance(
        extracted,
        list
    ):

        for item in extracted:

            if not isinstance(
                item,
                dict
            ):
                continue

            key = normalize_key(
                item.get("key")
            )

            value = clean_value(
                item.get("value")
            )

            if not key or not value:
                continue

            normalized.append(
                {
                    "key": key,
                    "value": value
                }
            )

    # --------------------------------------------------------
    # Fallback:
    #
    # {
    #   "loan_amount": "10 lakh"
    # }
    # --------------------------------------------------------

    elif isinstance(
        extracted,
        dict
    ):

        for key, value in extracted.items():

            key = normalize_key(
                key
            )

            value = clean_value(
                value
            )

            if not key or not value:
                continue

            normalized.append(
                {
                    "key": key,
                    "value": value
                }
            )

    return normalized


# ============================================================
# NORMALIZE MISSING INFORMATION
# ============================================================

def normalize_missing_information(
    missing: Any
) -> List[str]:

    normalized = []

    if isinstance(
        missing,
        list
    ):

        for item in missing:

            value = clean_value(
                item
            )

            if not value:
                continue

            # Prevent malformed dictionary strings.
            if (
                value.startswith("{")
                and value.endswith("}")
            ):
                continue

            normalized.append(
                value
            )

    elif isinstance(
        missing,
        str
    ):

        value = clean_value(
            missing
        )

        if value:
            normalized.append(
                value
            )

    return normalized


# ============================================================
# NORMALIZE LLM ANALYSIS
# ============================================================

def normalize_analysis(
    raw: Dict[str, Any]
) -> Dict[str, Any]:

    if not isinstance(
        raw,
        dict
    ):
        raw = {}

    result = (
        create_default_understanding()
    )

    # --------------------------------------------------------
    # Basic fields
    # --------------------------------------------------------

    string_fields = [

        "caller_name",

        "caller_role",

        "relationship_to_ram",

        "company_or_organization",

        "purpose",

        "domain",

        "topic",

        "rag_query",

        "reason",
    ]

    for field in string_fields:

        result[field] = clean_value(
            raw.get(field)
        )

    # --------------------------------------------------------
    # Relevance
    # --------------------------------------------------------

    relevance = clean_value(
        raw.get(
            "relevance"
        )
    )

    if relevance not in VALID_RELEVANCE:
        relevance = "UNKNOWN"

    result["relevance"] = relevance

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    safety = clean_value(
        raw.get(
            "safety"
        )
    )

    if safety not in VALID_SAFETY:
        safety = "SAFE"

    result["safety"] = safety

    # --------------------------------------------------------
    # Knowledge source
    # --------------------------------------------------------

    knowledge_source = clean_value(
        raw.get(
            "knowledge_source"
        )
    )

    if (
        knowledge_source
        not in VALID_KNOWLEDGE_SOURCES
    ):
        knowledge_source = "NONE"

    result[
        "knowledge_source"
    ] = knowledge_source

    # --------------------------------------------------------
    # Action
    # --------------------------------------------------------

    action = clean_value(
        raw.get(
            "action"
        )
    )

    if action not in VALID_ACTIONS:
        action = "CONTINUE"

    result["action"] = action

    # --------------------------------------------------------
    # Dynamic extracted information
    # --------------------------------------------------------

    result[
        "information_extracted"
    ] = normalize_information_extracted(
        raw.get(
            "information_extracted",
            []
        )
    )

    # --------------------------------------------------------
    # Missing information
    # --------------------------------------------------------

    result[
        "missing_information"
    ] = normalize_missing_information(
        raw.get(
            "missing_information",
            []
        )
    )

    return result


# ============================================================
# MERGE INFORMATION
# ============================================================

def merge_information(
    current: Dict[str, Any],
    understanding: Dict[str, Any]
) -> Dict[str, Any]:

    """
    Python owns canonical current-call memory.

    The LLM interprets the latest message.

    The LLM is NOT allowed to erase an established fact simply
    by returning null.
    """

    updated = dict(
        current
    )

    # --------------------------------------------------------
    # Basic fields
    # --------------------------------------------------------

    for field in BASIC_FIELDS:

        value = clean_value(
            understanding.get(
                field
            )
        )

        if value is not None:

            updated[field] = value

    # --------------------------------------------------------
    # Dynamic fields
    # --------------------------------------------------------

    extracted = understanding.get(
        "information_extracted",
        []
    )

    if isinstance(
        extracted,
        list
    ):

        for item in extracted:

            if not isinstance(
                item,
                dict
            ):
                continue

            key = normalize_key(
                item.get("key")
            )

            value = clean_value(
                item.get("value")
            )

            if not key or not value:
                continue

            updated[key] = value

    return updated


# ============================================================
# SYNCHRONIZE UNDERSTANDING WITH MEMORY
# ============================================================

def synchronize_with_memory(
    understanding: Dict[str, Any],
    information: Dict[str, Any]
) -> Dict[str, Any]:

    result = dict(
        understanding
    )

    # --------------------------------------------------------
    # Never forget established basic fields
    # --------------------------------------------------------

    for field in BASIC_FIELDS:

        if field in information:

            result[field] = (
                information[field]
            )

    # --------------------------------------------------------
    # Remove already-known missing information
    # --------------------------------------------------------

    missing = result.get(
        "missing_information",
        []
    )

    cleaned_missing = []

    known_keys = {
        normalize_key(key)
        for key in information.keys()
    }

    for item in missing:

        normalized_item = normalize_key(
            item
        )

        if (
            normalized_item
            and normalized_item in known_keys
        ):
            continue

        cleaned_missing.append(
            item
        )

    result[
        "missing_information"
    ] = cleaned_missing

    return result


# ============================================================
# FIND WHETHER CURRENT MEMORY ANSWERS A QUERY
# ============================================================

def memory_contains_requested_information(
    rag_query: Optional[str],
    information: Dict[str, Any]
) -> bool:

    if not rag_query:
        return False

    query = rag_query.lower()

    # --------------------------------------------------------
    # Exact / normalized key matching
    # --------------------------------------------------------

    for key in information.keys():

        normalized_key = (
            str(key)
            .lower()
            .replace("_", " ")
        )

        if normalized_key in query:
            return True

    # --------------------------------------------------------
    # Common semantic aliases
    # --------------------------------------------------------

    aliases = {

        "processing fee": [
            "processing_fee",
            "fee",
        ],

        "interest rate": [
            "interest_rate",
            "rate",
        ],

        "loan amount": [
            "loan_amount",
            "amount",
        ],

        "loan tenure": [
            "loan_tenure",
            "tenure",
        ],

        "tracking number": [
            "tracking_number",
            "tracking",
        ],

        "delivery date": [
            "expected_delivery",
            "delivery_date",
        ],
    }

    for phrase, keys in aliases.items():

        if phrase not in query:
            continue

        for key in keys:

            if key in information:
                return True

    return False


# ============================================================
# SEMANTIC CONSISTENCY
# ============================================================

def enforce_semantic_consistency(
    state: ConversationState,
    understanding: Dict[str, Any],
    information: Dict[str, Any]
) -> Dict[str, Any]:

    result = synchronize_with_memory(
        understanding,
        information
    )

    action = result.get(
        "action",
        "CONTINUE"
    )

    knowledge_source = result.get(
        "knowledge_source",
        "NONE"
    )

    purpose = information.get(
        "purpose"
    )

    relevance = result.get(
        "relevance",
        "UNKNOWN"
    )

    # --------------------------------------------------------
    # Current-call memory always wins
    # --------------------------------------------------------

    if knowledge_source == "ASSISTANT_KB":

        rag_query = result.get(
            "rag_query"
        )

        if memory_contains_requested_information(
            rag_query,
            information
        ):

            result[
                "knowledge_source"
            ] = "CURRENT_CALL_MEMORY"

            result[
                "rag_query"
            ] = None

            result[
                "action"
            ] = "CONTINUE"

            result[
                "reason"
            ] = (
                "The requested information is already "
                "available in current-call memory."
            )

    # --------------------------------------------------------
    # If caller owns missing information, don't force RAG
    # --------------------------------------------------------

    if (
        knowledge_source == "ASSISTANT_KB"
        and purpose is not None
        and result.get(
            "missing_information"
        )
    ):

        # We only override RAG when the LLM has indicated that
        # the information belongs to the caller.
        #
        # This is communicated by the knowledge source itself
        # on normal turns, so we don't blindly replace every RAG
        # decision here.
        pass

    # --------------------------------------------------------
    # Known purpose must remain known
    # --------------------------------------------------------

    if (
        purpose is not None
        and result.get(
            "purpose"
        ) is None
    ):

        result[
            "purpose"
        ] = purpose

    # --------------------------------------------------------
    # Known information cannot remain "missing"
    # --------------------------------------------------------

    known_keys = {
        normalize_key(key)
        for key in information.keys()
    }

    filtered_missing = []

    for item in result.get(
        "missing_information",
        []
    ):

        normalized_item = normalize_key(
            item
        )

        if (
            normalized_item
            and normalized_item in known_keys
        ):
            continue

        filtered_missing.append(
            item
        )

    result[
        "missing_information"
    ] = filtered_missing

    # --------------------------------------------------------
    # CLARIFY is only for unknown purpose
    # --------------------------------------------------------

    if (
        result.get(
            "action"
        ) == "CLARIFY"
        and purpose is not None
        and relevance != "UNKNOWN"
    ):

        result[
            "action"
        ] = "CONTINUE"

        result[
            "reason"
        ] = (
            "The purpose is already known. "
            "Continue the conversation naturally."
        )

    # --------------------------------------------------------
    # If caller explicitly wants to end, allow termination
    # --------------------------------------------------------

    if action == "TERMINATE":
        result["action"] = "TERMINATE"

    return result


# ============================================================
# ANALYSIS PROMPT
# ============================================================

def build_analysis_prompt(
    state: ConversationState
) -> str:

    history = format_history(
        state[
            "conversation_history"
        ]
    )

    current_information = json.dumps(
        state[
            "information_collected"
        ],
        indent=2,
        ensure_ascii=False
    )

    latest_message = state[
        "user_message"
    ]

    clarification_count = state[
        "clarification_count"
    ]

    return f"""
You are the semantic intelligence engine of Ram's personal
AI Call Assistant.

You are handling ONE CURRENT PHONE CALL.

The assistant represents Ram.

The assistant is NOT Ram.

The person speaking is the CALLER.

Your task is to understand the caller semantically, preserve
conversation memory, determine information ownership, decide
whether RAG is needed, and select the correct action.

============================================================
CURRENT CALL MEMORY
============================================================

{current_information}

============================================================
PREVIOUS CONVERSATION
============================================================

{history}

============================================================
LATEST CALLER MESSAGE
============================================================

{latest_message}

============================================================
CLARIFICATION COUNT
============================================================

{clarification_count}

Maximum purpose clarifications:
{MAX_CLARIFICATIONS}

============================================================
MOST IMPORTANT RULE
============================================================

UNDERSTAND MEANING.

Do NOT depend on exact sentences.

Do NOT require predefined wording.

Do NOT assume the domain.

Do NOT assume the purpose.

Infer meaning from the conversation.

============================================================
CALLER IDENTITY
============================================================

Keep these concepts separate:

caller_name:
The actual person's name, if known.

caller_role:
The person's role, if known.

company_or_organization:
The organization they represent, if any.

relationship_to_ram:
Their personal relationship to Ram, if applicable.

Example:

"I'm calling from ABC Finance."

Correct:

company_or_organization = "ABC Finance"

caller_name = null

caller_role may remain null

purpose = null

The organization does NOT automatically determine the purpose.

============================================================
PURPOSE
============================================================

Determine why the caller is contacting Ram.

Examples:

loan offer
job opportunity
package delivery
telecom service
college matter
personal message
appointment
sales call

These are examples only.

Do not restrict the system to these categories.

============================================================
DYNAMIC INFORMATION EXTRACTION
============================================================

Extract useful facts from the latest caller message.

Use dynamic key/value pairs.

Example:

[
    {{"key": "loan_amount", "value": "10 lakh"}},
    {{"key": "interest_rate", "value": "11 percent"}},
    {{"key": "loan_tenure", "value": "5 years"}}
]

Other examples:

tracking_number
expected_delivery
job_role
interview_date
appointment_time
caller_message
reference_id
service_name
product_name

These are examples only.

Do not restrict the system to these keys.

Extract only facts actually supported by the caller.

Do not invent values.

============================================================
MEMORY
============================================================

CURRENT CALL MEMORY IS CANONICAL.

Never erase established information because the latest LLM
output returns null.

Never ask for information that is already known.

Example:

Earlier:

"The processing fee is 2 percent."

Later:

"What is the processing fee?"

Correct:

knowledge_source = CURRENT_CALL_MEMORY

Do not ask again.

============================================================
ANSWER SOURCE PRIORITY
============================================================

Use this conceptual priority:

1. CURRENT_CALL_MEMORY
2. CALLER
3. ASSISTANT_KB
4. NONE

============================================================
CURRENT_CALL_MEMORY
============================================================

Use when the answer was already given during this call.

============================================================
CALLER
============================================================

Use when the caller naturally owns the information.

Example:

A bank representative is discussing THEIR offer.

If the assistant asks:

"What is the processing fee for this offer?"

and the caller has not provided it yet:

knowledge_source = CALLER

The assistant should ask the caller.

Do NOT substitute a generic knowledge-base value.

============================================================
ASSISTANT_KB
============================================================

Use only when assistant-owned approved knowledge is genuinely
needed.

Examples:

- documented assistant policy
- approved general product information
- approved internal instructions

If ASSISTANT_KB is selected:

rag_query must contain the actual knowledge question.

============================================================
RAG SAFETY
============================================================

Do NOT use generic RAG to answer a caller's specific offer,
specific delivery, specific appointment, specific salary,
specific application, or other caller-owned information.

Example:

Caller says:

"Our processing fee for this offer is 1 percent."

That 1 percent is caller-owned current-call information.

Use memory.

Do not replace it with:

"Up to 2 percent"

from the generic knowledge base.

============================================================
FINANCIAL SAFETY
============================================================

The assistant MAY receive and record:

- loan amount
- interest rate
- tenure
- processing fee
- charges
- financial offer details

Receiving an offer is NOT a commitment.

The assistant MUST NOT:

- accept an offer
- reject an offer
- approve anything
- promise anything
- negotiate
- confirm acceptance
- make a financial decision
- commit Ram

Example:

"We can offer 10 lakh at 11 percent for five years."

Action:

CONTINUE

But:

"Can you confirm Ram accepts the offer?"

Action:

REFUSE_COMMITMENT

============================================================
CLARIFICATION
============================================================

Use CLARIFY ONLY when the overall purpose is genuinely unknown.

There is a maximum of one purpose clarification.

Do NOT use CLARIFY merely because an optional detail is missing.

If purpose is known, continue naturally and ask for relevant
missing information when appropriate.

============================================================
TOPIC CHANGES
============================================================

The caller can change topics.

Do not erase previous information.

Preserve old facts while understanding the new topic.

============================================================
PERSONAL CALLS
============================================================

Do not force personal callers through business questions.

Example:

"I'm Ram's sister."

This establishes:

relationship_to_ram = sister

A natural next response could ask:

"What would you like me to tell Ram?"

============================================================
IRRELEVANCE
============================================================

Do not call something irrelevant merely because the domain is
unknown.

Use IRRELEVANT only when the conversation is clearly unrelated
to Ram and has no meaningful reason to continue.

============================================================
THREATS / INAPPROPRIATE
============================================================

Threatening, abusive, harassing, or clearly inappropriate calls
should not continue.

============================================================
EXTERNAL ACTIONS
============================================================

The following are NOT currently implemented:

- sending email
- sending notifications
- making another call
- checking live delivery systems
- updating databases
- scheduling reminders
- booking appointments
- checking bank systems

Do NOT claim these actions happened.

You may say:

"I've noted that information."

"I'll pass that information along to Ram."

============================================================
ACTIONS
============================================================

CONTINUE:
Continue naturally.

CLARIFY:
Purpose genuinely unknown and one clarification is needed.

TERMINATE:
End normally.

TERMINATE_SENSITIVE:
Sensitive information requires termination.

REFUSE_COMMITMENT:
Caller asks the assistant to make or confirm a decision or
commitment on Ram's behalf.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use this structure:

{{
    "caller_name": null,
    "caller_role": null,
    "relationship_to_ram": null,
    "company_or_organization": null,
    "purpose": null,
    "domain": null,
    "topic": null,
    "relevance": "UNKNOWN",
    "safety": "SAFE",
    "information_extracted": [],
    "missing_information": [],
    "knowledge_source": "NONE",
    "rag_query": null,
    "action": "CONTINUE",
    "reason": ""
}}

Do not return markdown.

Do not return explanation outside JSON.
"""


# ============================================================
# ANALYZE CALL
# ============================================================

def analyze_call(
    state: ConversationState
) -> Dict[str, Any]:

    messages = [

        {
            "role": "system",

            "content": (
                "You are the semantic intelligence engine "
                "for Ram's personal AI Call Assistant. "
                "Understand meaning, preserve memory, "
                "determine information ownership, and "
                "return valid JSON only."
            )
        },

        {
            "role": "user",

            "content": build_analysis_prompt(
                state
            )
        }
    ]

    raw = call_groq_json(
        messages
    )

    return normalize_analysis(
        raw
    )


# ============================================================
# RAG CONTEXT FORMATTER
# ============================================================

def format_rag_context(
    chunks: List[Dict[str, Any]]
) -> str:

    if not chunks:
        return ""

    parts = []

    for chunk in chunks:

        source = chunk.get(
            "source",
            "unknown"
        )

        text = chunk.get(
            "text",
            ""
        )

        if text:

            parts.append(
                f"SOURCE: {source}\n{text}"
            )

    return (
        "\n\n--------------------\n\n"
        .join(parts)
    )


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_rag_context(
    understanding: Dict[str, Any]
) -> str:

    if (
        understanding.get(
            "knowledge_source"
        )
        != "ASSISTANT_KB"
    ):
        return ""

    rag_query = understanding.get(
        "rag_query"
    )

    if not rag_query:
        return ""

    try:

        chunks = retrieve_context(
            rag_query,
            top_k=3
        )

        return format_rag_context(
            chunks
        )

    except Exception as error:

        print(
            f"\nRAG retrieval warning: {error}"
        )

        return ""


# ============================================================
# RESPONSE GENERATION
# ============================================================

def generate_response(
    state: ConversationState,
    understanding: Dict[str, Any],
    rag_context: str
) -> str:

    history = format_history(
        state[
            "conversation_history"
        ]
    )

    information = json.dumps(
        state[
            "information_collected"
        ],
        indent=2,
        ensure_ascii=False
    )

    latest_message = state[
        "user_message"
    ]

    action = understanding.get(
        "action",
        "CONTINUE"
    )

    knowledge_source = understanding.get(
        "knowledge_source",
        "NONE"
    )

    missing_information = json.dumps(
        understanding.get(
            "missing_information",
            []
        ),
        ensure_ascii=False
    )

    rag_instruction = ""

    # --------------------------------------------------------
    # RAG context
    # --------------------------------------------------------

    if knowledge_source == "ASSISTANT_KB":

        if rag_context:

            rag_instruction = f"""
============================================================
APPROVED KNOWLEDGE BASE
============================================================

{rag_context}

Use this context only for assistant-owned knowledge.

Do not invent facts beyond the supplied context.

If the supplied context does not answer the question,
say naturally that you do not have that information.
"""

        else:

            rag_instruction = """
============================================================
APPROVED KNOWLEDGE BASE
============================================================

No sufficiently relevant approved knowledge-base information
was found.

Do NOT guess.

Say naturally that you do not have that information.
"""

    # --------------------------------------------------------
    # Response prompt
    # --------------------------------------------------------

    response_prompt = f"""
You are the spoken-response layer of Ram's personal AI
Call Assistant.

The caller is speaking to you right now.

You represent Ram.

You are NOT Ram.

============================================================
CURRENT CONVERSATION
============================================================

{history}

============================================================
CANONICAL CURRENT-CALL MEMORY
============================================================

{information}

============================================================
LATEST CALLER MESSAGE
============================================================

{latest_message}

============================================================
INTERNAL ANALYSIS
============================================================

Caller name:
{understanding.get("caller_name")}

Caller role:
{understanding.get("caller_role")}

Relationship:
{understanding.get("relationship_to_ram")}

Company:
{understanding.get("company_or_organization")}

Purpose:
{understanding.get("purpose")}

Domain:
{understanding.get("domain")}

Topic:
{understanding.get("topic")}

Action:
{action}

Knowledge source:
{knowledge_source}

Missing information:
{missing_information}

Reason:
{understanding.get("reason", "")}

{rag_instruction}

============================================================
MEMORY RULE
============================================================

CANONICAL CURRENT-CALL MEMORY IS AUTHORITATIVE.

If the answer is already in memory, answer directly.

Do NOT ask for it again.

Do NOT replace it with generic RAG information.

============================================================
CALLER-OWNED INFORMATION
============================================================

If the caller naturally owns the missing information,
ask the caller for it.

Example:

Caller is a bank representative discussing their own offer.

The processing fee has not yet been provided.

Ask naturally:

"Could you let me know the processing fee for this offer?"

Do not use a generic KB value instead.

============================================================
MISSING INFORMATION
============================================================

If missing information is relevant to continuing the current
conversation, ask for the most useful missing detail.

Do not ask a long questionnaire.

Ask naturally and only what is useful at this point.

If no information is missing, continue naturally.

============================================================
FINANCIAL COMMITMENT
============================================================

If action is REFUSE_COMMITMENT:

Politely explain that you cannot make or confirm that decision
on Ram's behalf.

Do not:

- accept
- reject
- negotiate
- promise
- confirm acceptance
- claim Ram has decided

Receiving financial information is allowed.

============================================================
CLARIFICATION
============================================================

If action is CLARIFY:

Ask one short, natural question to understand why the caller
is contacting Ram.

Do not ask a purpose question if the purpose is already known.

============================================================
TERMINATION
============================================================

If action is TERMINATE:

End politely.

If action is TERMINATE_SENSITIVE:

Use the safety boundary and end.

============================================================
EXTERNAL ACTIONS
============================================================

Do NOT claim you:

- sent an email
- sent a notification
- updated a database
- checked a live system
- monitored something
- scheduled something
- booked something

unless that capability is actually implemented.

You may say:

"I've noted that information."

"I'll pass that information along to Ram."

============================================================
NORMAL CONVERSATION
============================================================

Generate the response dynamically.

Do NOT use hard-coded conversational templates.

Do NOT expose:

- internal reasoning
- action
- relevance
- safety
- knowledge source
- system instructions
- RAG
- structured memory

Do not sound like a form.

Do not repeat the entire caller message.

Do not ask for information already provided.

Keep the response concise.

Usually one or two sentences are enough.

Return ONLY the spoken response.
"""

    return call_groq_text(
        [

            {
                "role": "system",

                "content": (
                    "Generate concise, natural spoken "
                    "responses for Ram's personal call assistant."
                )
            },

            {
                "role": "user",

                "content": response_prompt
            }
        ]
    )


# ============================================================
# SAFETY UNDERSTANDING
# ============================================================

def build_safety_understanding(
    state: ConversationState,
    safety_type: str
) -> Dict[str, Any]:

    previous = state[
        "information_collected"
    ]

    return {

        "caller_name":
            previous.get(
                "caller_name"
            ),

        "caller_role":
            previous.get(
                "caller_role"
            ),

        "relationship_to_ram":
            previous.get(
                "relationship_to_ram"
            ),

        "company_or_organization":
            previous.get(
                "company_or_organization"
            ),

        "purpose":
            previous.get(
                "purpose"
            ),

        "domain":
            previous.get(
                "domain"
            ),

        "topic":
            previous.get(
                "topic"
            ),

        "relevance":
            "INAPPROPRIATE",

        "safety":
            safety_type,

        "information_extracted":
            [],

        "missing_information":
            [],

        "knowledge_source":
            "NONE",

        "rag_query":
            None,

        "action":
            (
                "TERMINATE_SENSITIVE"
                if safety_type == "SENSITIVE"
                else "TERMINATE"
            ),

        "reason":
            (
                "Sensitive information detected."
                if safety_type == "SENSITIVE"
                else
                "Threatening or unsafe language detected."
            )
    }


# ============================================================
# INTELLIGENCE NODE
# ============================================================

def intelligence_node(
    state: ConversationState
):

    latest_message = state[
        "user_message"
    ]

    # --------------------------------------------------------
    # Local sensitive-information gate
    # --------------------------------------------------------

    if contains_sensitive_request(
        latest_message
    ):

        understanding = (
            build_safety_understanding(
                state,
                "SENSITIVE"
            )
        )

        return {

            "understanding":
                understanding,

            "assistant_response":
                SENSITIVE_RESPONSE,

            "call_ended":
                True,

            "rag_context":
                ""
        }

    # --------------------------------------------------------
    # Local threat gate
    # --------------------------------------------------------

    if contains_threat(
        latest_message
    ):

        understanding = (
            build_safety_understanding(
                state,
                "UNSAFE"
            )
        )

        return {

            "understanding":
                understanding,

            "assistant_response":
                THREAT_RESPONSE,

            "call_ended":
                True,

            "rag_context":
                ""
        }

    # --------------------------------------------------------
    # LLM semantic understanding
    # --------------------------------------------------------

    try:

        understanding = analyze_call(
            state
        )

    except Exception as error:

        print(
            f"\nLLM analysis error: {error}"
        )

        fallback = (
            create_default_understanding()
        )

        # Preserve existing memory.
        for field in BASIC_FIELDS:

            fallback[field] = (
                state[
                    "information_collected"
                ].get(field)
            )

        fallback[
            "action"
        ] = "TERMINATE"

        fallback[
            "reason"
        ] = (
            "Unable to safely analyze the call."
        )

        return {

            "understanding":
                fallback,

            "assistant_response":
                LLM_FAILURE_RESPONSE,

            "call_ended":
                True,

            "rag_context":
                ""
        }

    # --------------------------------------------------------
    # LLM safety result
    # --------------------------------------------------------

    if (
        understanding.get(
            "safety"
        )
        == "SENSITIVE"
    ):

        understanding[
            "action"
        ] = "TERMINATE_SENSITIVE"

        return {

            "understanding":
                understanding,

            "assistant_response":
                SENSITIVE_RESPONSE,

            "call_ended":
                True,

            "rag_context":
                ""
        }

    if (
        understanding.get(
            "safety"
        )
        == "UNSAFE"
    ):

        understanding[
            "action"
        ] = "TERMINATE"

        return {

            "understanding":
                understanding,

            "assistant_response":
                THREAT_RESPONSE,

            "call_ended":
                True,

            "rag_context":
                ""
        }

    return {
        "understanding":
            understanding
    }


# ============================================================
# INFORMATION NODE
# ============================================================

def information_node(
    state: ConversationState
):

    current = state[
        "information_collected"
    ]

    understanding = state[
        "understanding"
    ]

    # --------------------------------------------------------
    # Merge latest semantic information
    # --------------------------------------------------------

    updated = merge_information(
        current,
        understanding
    )

    # --------------------------------------------------------
    # Protect state
    # --------------------------------------------------------

    normalized = (
        enforce_semantic_consistency(
            state,
            understanding,
            updated
        )
    )

    return {

        "information_collected":
            updated,

        "understanding":
            normalized
    }


# ============================================================
# RESPONSE NODE
# ============================================================

def response_node(
    state: ConversationState
):

    understanding = state[
        "understanding"
    ]

    # --------------------------------------------------------
    # Safety already generated response
    # --------------------------------------------------------

    if state[
        "call_ended"
    ]:

        return {}

    action = understanding.get(
        "action",
        "CONTINUE"
    )

    relevance = understanding.get(
        "relevance",
        "UNKNOWN"
    )

    # --------------------------------------------------------
    # IRRELEVANT
    # --------------------------------------------------------

    if relevance == "IRRELEVANT":

        understanding[
            "action"
        ] = "TERMINATE"

        return {

            "understanding":
                understanding,

            "assistant_response":
                IRRELEVANT_RESPONSE,

            "call_ended":
                True,

            "rag_context":
                ""
        }

    # --------------------------------------------------------
    # INAPPROPRIATE
    # --------------------------------------------------------

    if relevance == "INAPPROPRIATE":

        understanding[
            "action"
        ] = "TERMINATE"

        return {

            "understanding":
                understanding,

            "assistant_response":
                THREAT_RESPONSE,

            "call_ended":
                True,

            "rag_context":
                ""
        }

    # --------------------------------------------------------
    # CLARIFY
    # --------------------------------------------------------

    if action == "CLARIFY":

        purpose = understanding.get(
            "purpose"
        )

        purpose_unknown = (
            purpose is None
            or relevance == "UNKNOWN"
        )

        if purpose_unknown:

            if (
                state[
                    "clarification_count"
                ]
                >= MAX_CLARIFICATIONS
            ):

                understanding[
                    "action"
                ] = "TERMINATE"

                return {

                    "understanding":
                        understanding,

                    "assistant_response":
                        CLARIFICATION_LIMIT_RESPONSE,

                    "call_ended":
                        True,

                    "rag_context":
                        ""
                }

    # --------------------------------------------------------
    # TERMINATE
    # --------------------------------------------------------

    if action == "TERMINATE":

        try:

            response = generate_response(
                state,
                understanding,
                ""
            )

        except Exception:

            response = (
                "Thank you for calling. Goodbye."
            )

        return {

            "understanding":
                understanding,

            "assistant_response":
                response,

            "call_ended":
                True,

            "rag_context":
                ""
        }

    # --------------------------------------------------------
    # TERMINATE SENSITIVE
    # --------------------------------------------------------

    if action == "TERMINATE_SENSITIVE":

        return {

            "understanding":
                understanding,

            "assistant_response":
                SENSITIVE_RESPONSE,

            "call_ended":
                True,

            "rag_context":
                ""
        }

    # --------------------------------------------------------
    # REFUSE COMMITMENT
    # --------------------------------------------------------

    if action == "REFUSE_COMMITMENT":

        try:

            response = generate_response(
                state,
                understanding,
                ""
            )

        except Exception:

            response = (
                "I'm not able to make or confirm "
                "that decision on Ram's behalf."
            )

        return {

            "understanding":
                understanding,

            "assistant_response":
                response,

            "call_ended":
                False,

            "rag_context":
                ""
        }

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    rag_context = ""

    if (
        understanding.get(
            "knowledge_source"
        )
        == "ASSISTANT_KB"
    ):

        rag_context = (
            retrieve_rag_context(
                understanding
            )
        )

    # --------------------------------------------------------
    # NORMAL RESPONSE
    # --------------------------------------------------------

    try:

        response = generate_response(
            state,
            understanding,
            rag_context
        )

    except Exception as error:

        print(
            f"\nResponse generation error: {error}"
        )

        response = (
            "I'm sorry, could you tell me "
            "a little more about that?"
        )

    if not response.strip():

        response = (
            "I'm sorry, could you tell me "
            "a little more about that?"
        )

    return {

        "understanding":
            understanding,

        "assistant_response":
            response,

        "call_ended":
            False,

        "rag_context":
            rag_context
    }


# ============================================================
# CLARIFICATION NODE
# ============================================================

def clarification_node(
    state: ConversationState
):

    action = state[
        "understanding"
    ].get(
        "action",
        "CONTINUE"
    )

    if action == "CLARIFY":

        return {

            "clarification_count":
                (
                    state[
                        "clarification_count"
                    ]
                    + 1
                )
        }

    return {}


# ============================================================
# LANGGRAPH
# ============================================================

def build_graph():

    graph = StateGraph(
        ConversationState
    )

    graph.add_node(
        "intelligence",
        intelligence_node
    )

    graph.add_node(
        "information",
        information_node
    )

    graph.add_node(
        "response",
        response_node
    )

    graph.add_node(
        "clarification",
        clarification_node
    )

    graph.set_entry_point(
        "intelligence"
    )

    graph.add_edge(
        "intelligence",
        "information"
    )

    graph.add_edge(
        "information",
        "response"
    )

    graph.add_edge(
        "response",
        "clarification"
    )

    graph.add_edge(
        "clarification",
        END
    )

    return graph.compile()


# ============================================================
# DISPLAY UNDERSTANDING
# ============================================================

def display_understanding(
    understanding: Dict[str, Any]
):

    print()

    print(
        "Understanding:"
    )

    print(
        json.dumps(
            understanding,
            indent=2,
            ensure_ascii=False
        )
    )


# ============================================================
# DISPLAY INFORMATION
# ============================================================

def display_information(
    information: Dict[str, Any]
):

    print()

    print(
        "Information collected:"
    )

    print(
        json.dumps(
            information,
            indent=2,
            ensure_ascii=False
        )
    )

# ============================================================
# STRUCTURED CALL SUMMARY + LOCAL ARTIFACTS
# ============================================================

def generate_call_summary(
    state: ConversationState,
    end_reason: str
) -> Dict[str, Any]:
    """Generate a generic structured summary using Groq."""

    information = state.get("information_collected", {})
    understanding = state.get("understanding", {})
    transcript = "\n".join(
        state.get("conversation_history", [])
    )

    prompt = f"""
You are summarizing a completed call for Ram's personal
AI Call Assistant.

Analyze the actual conversation and produce a factual,
concise, domain-independent summary.

Do not assume the call is about finance or loans.
It could involve recruitment, delivery, education,
personal matters, services, or any other topic.

Rules:
- Use only information supported by the transcript or memory.
- Do not invent names, facts, promises, or outcomes.
- Use null for unknown personal or organizational details.
- Extract useful facts as dynamic key/value pairs.
- Action items must be sensible next steps for Ram.
- Do not claim an email, notification, or external action occurred.
- Clearly indicate whether the assistant made a commitment.
- The assistant must not make commitments on Ram's behalf.
- Do not include the transcript in the summary JSON.
- Return valid JSON only.

CALL END REASON:
{end_reason}

CURRENT-CALL MEMORY:
{json.dumps(information, ensure_ascii=False, indent=2)}

LATEST UNDERSTANDING:
{json.dumps(understanding, ensure_ascii=False, indent=2)}

FULL TRANSCRIPT:
{transcript}

Return exactly these top-level fields:
{{
    "caller_name": null,
    "caller_role": null,
    "company_or_organization": null,
    "relationship_to_ram": null,
    "purpose": null,
    "domain": null,
    "key_information": [],
    "outcome": "",
    "action_items": [],
    "commitment_status": "",
    "termination_reason": ""
}}

For key_information, use objects containing "key" and "value".
Choose keys appropriate to this particular call.
"""

    try:
        raw = call_groq_json([
            {
                "role": "system",
                "content": (
                    "Create a factual, generic call summary. "
                    "Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ])
        generated_by = "groq"

    except Exception as error:
        print(f"\nSummary generation warning: {error}")
        raw = {}
        generated_by = "fallback"

    if not isinstance(raw, dict):
        raw = {}
        generated_by = "fallback"

    def summary_string(key: str) -> Optional[str]:
        value = clean_value(raw.get(key))

        if value is not None:
            return value

        return clean_value(information.get(key))

    key_information = normalize_information_extracted(
        raw.get("key_information", [])
    )

    # Preserve facts already recorded by the conversation engine.
    basic_fields = set(BASIC_FIELDS)
    existing_keys = {
        item["key"] for item in key_information
    }

    for key, value in information.items():
        normalized_key = normalize_key(key)
        cleaned_value = clean_value(value)

        if (
            normalized_key
            and normalized_key not in basic_fields
            and cleaned_value
            and normalized_key not in existing_keys
        ):
            key_information.append({
                "key": normalized_key,
                "value": cleaned_value
            })
            existing_keys.add(normalized_key)

    action_items_raw = raw.get("action_items", [])
    action_items = []

    if isinstance(action_items_raw, list):
        for item in action_items_raw:
            cleaned = clean_value(item)
            if cleaned:
                action_items.append(cleaned)

    elif isinstance(action_items_raw, str):
        cleaned = clean_value(action_items_raw)
        if cleaned:
            action_items.append(cleaned)

    return {
        "caller_name": summary_string("caller_name"),
        "caller_role": summary_string("caller_role"),
        "company_or_organization": summary_string(
            "company_or_organization"
        ),
        "relationship_to_ram": summary_string(
            "relationship_to_ram"
        ),
        "purpose": summary_string("purpose"),
        "domain": summary_string("domain"),
        "key_information": key_information,
        "outcome": (
            clean_value(raw.get("outcome"))
            or "Call ended. Review the transcript for details."
        ),
        "action_items": action_items,
        "commitment_status": (
            clean_value(raw.get("commitment_status"))
            or "Not established by the summary model."
        ),
        "termination_reason": (
            clean_value(raw.get("termination_reason"))
            or end_reason
        ),
        "summary_generated_by": generated_by
    }


def save_call_artifacts(
    state: ConversationState,
    end_reason: str
) -> Optional[str]:
    """Save a call summary and transcript to unique local files."""

    transcript_lines = list(
        state.get("conversation_history", [])
    )

    if not transcript_lines:
        print(
            "No conversation was recorded; "
            "no call summary file was created."
        )
        return None

    summary = generate_call_summary(state, end_reason)

    call_time = datetime.now().astimezone()
    call_id = call_time.strftime("%Y%m%d_%H%M%S_%f")

    output_dir = os.path.join(BASE_DIR, "call_logs")
    os.makedirs(output_dir, exist_ok=True)

    transcript = "\n".join(transcript_lines)

    payload = {
        "call_id": call_id,
        "created_at": call_time.isoformat(timespec="seconds"),
        "end_reason": end_reason,
        "summary": summary,
        "transcript": transcript_lines
    }

    json_path = os.path.join(
        output_dir,
        f"call_{call_id}.json"
    )

    txt_path = os.path.join(
        output_dir,
        f"call_{call_id}.txt"
    )

    # Machine-readable record.
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2
        )

    # Human-readable record.
    readable_lines = [
        "AI PERSONAL CALL ASSISTANT — CALL SUMMARY",
        "=" * 48,
        f"Created at: {payload['created_at']}",
        f"End reason: {end_reason}",
        "",
        f"Caller: {summary['caller_name'] or 'Unknown'}",
        f"Role: {summary['caller_role'] or 'Unknown'}",
        (
            "Organization: "
            f"{summary['company_or_organization'] or 'Unknown'}"
        ),
        (
            "Relationship to Ram: "
            f"{summary['relationship_to_ram'] or 'Unknown'}"
        ),
        f"Purpose: {summary['purpose'] or 'Unknown'}",
        f"Domain: {summary['domain'] or 'Unknown'}",
        "",
        "Key information:"
    ]

    if summary["key_information"]:
        for item in summary["key_information"]:
            label = item["key"].replace("_", " ").title()
            readable_lines.append(
                f"- {label}: {item['value']}"
            )
    else:
        readable_lines.append(
            "- No additional information recorded."
        )

    readable_lines.extend([
        "",
        f"Outcome: {summary['outcome']}",
        f"Commitment status: {summary['commitment_status']}",
        f"Termination reason: {summary['termination_reason']}",
        "",
        "Action items:"
    ])

    if summary["action_items"]:
        for item in summary["action_items"]:
            readable_lines.append(f"- {item}")
    else:
        readable_lines.append("- None identified.")

    readable_lines.extend([
        "",
        "FULL TRANSCRIPT",
        "=" * 48,
        transcript,
        ""
    ])

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write("\n".join(readable_lines))

    print("\nSTRUCTURED CALL SUMMARY")
    print("=" * 48)
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        )
    )

    print("\nSaved call files:")
    print(f"- {json_path}")
    print(f"- {txt_path}")

    return json_path


# ============================================================
# RUN ONE CALL
# ============================================================

def run_call():
    print("\n" + "=" * 70)
    print("              AI PERSONAL CALL ASSISTANT")
    print("           GENERIC LLM CONVERSATION ENGINE")
    print("=" * 70)
    print("This execution represents ONE completely new call.")
    print("All conversational memory is limited to this call.")
    print("Type 'exit' to manually end the call.\n")

    # Every execution starts with fresh memory.
    state = create_initial_state()
    end_reason = "Call ended."

    try:
        graph = build_graph()

        while not state["call_ended"]:
            try:
                caller_message = input("Caller: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nCall interrupted by the user.")
                end_reason = "Caller ended the session manually."
                break

            if not caller_message:
                continue

            if caller_message.lower() in {"exit", "quit", "bye"}:
                goodbye = "Thank you for calling. Have a great day!"
                print(f"\nAI: {goodbye}")
                state["conversation_history"].append(f"Caller: {caller_message}")
                state["conversation_history"].append(f"AI: {goodbye}")
                end_reason = "Caller ended the session manually."
                break

            state["user_message"] = caller_message
            state["conversation_history"].append(f"Caller: {caller_message}")
            state["turn_count"] += 1

            try:
                result = graph.invoke(state)
                state.update(result)
            except Exception as error:
                print(f"\nUnexpected application error: {error}")
                end_reason = f"Application error: {type(error).__name__}"
                break

            display_understanding(state["understanding"])

            assistant_response = state.get("assistant_response") or "Thank you for calling. Goodbye."
            print(f"\nAI: {assistant_response}")
            state["conversation_history"].append(f"AI: {assistant_response}")

            if state["call_ended"]:
                end_reason = (
                    state.get("understanding", {}).get("reason")
                    or state.get("understanding", {}).get("action")
                    or "Assistant ended the call."
                )
                print("\n" + "=" * 70)
                print("                         CALL ENDED")
                print("=" * 70)
                display_information(state["information_collected"])
                break

    except Exception as error:
        print(f"\nCall setup error: {error}")
        end_reason = f"Call setup error: {type(error).__name__}"

    
    finally:
        # Save the call report before attempting email delivery.
        try:
            json_path = save_call_artifacts(state, end_reason)

            if json_path:
                try:
                    from email_service import send_call_email
                    send_call_email(json_path)
                except Exception as error:
                    print(f"\nEmail delivery failed: {error}")
                    print("The call report remains saved locally.")
            else:
                print("\nNo call transcript available; email skipped.")

        except Exception as error:
            print(f"\nCould not save call artifacts: {error}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_call()
