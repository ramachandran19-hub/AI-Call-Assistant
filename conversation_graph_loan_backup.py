"""
AI PERSONAL CALL ASSISTANT

The caller is the bank / loan agent.

The AI represents Ram as his personal call assistant.

Main responsibility:
- Understand why the caller is calling
- Collect loan/offer details FROM THE CALLER
- Ask naturally for missing details
- Use RAG only when the caller genuinely asks
  for information from the assistant's knowledge base
- Never invent missing loan terms
- Never approve or accept a loan on Ram's behalf
- Eventually pass the collected information to Ram
"""

import os
import json
import re
from typing import TypedDict, Dict, List, Any

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END

from rag import retrieve_context


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(
    os.path.join(BASE_DIR, ".env")
)

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Groq API key not found.\n"
        "Add GROQ_API_KEY=your_key to .env"
    )

client = Groq(
    api_key=API_KEY
)

MODEL = "openai/gpt-oss-20b"


# ============================================================
# CALL STATE
# ============================================================

class ConversationState(TypedDict):

    user_message: str

    conversation_history: List[str]

    information_collected: Dict[str, Any]

    understanding: Dict[str, Any]

    assistant_response: str

    action: str

    call_ended: bool

    turn_count: int

    rag_context: List[Dict[str, Any]]


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
        "action": "",
        "call_ended": False,
        "turn_count": 0,
        "rag_context": [],
    }


# ============================================================
# LOCAL SAFETY CHECKS
# ============================================================

SENSITIVE_PATTERNS = [

    r"\botp\b",
    r"\bone time password\b",
    r"\bpassword\b",
    r"\bpin\b",
    r"\bmpin\b",
    r"\bcvv\b",
    r"\bcard number\b",
    r"\bdebit card number\b",
    r"\bcredit card number\b",

]


THREAT_PATTERNS = [

    r"\bkill you\b",
    r"\bi will kill\b",
    r"\bkill him\b",
    r"\bkill her\b",
    r"\bshoot you\b",
    r"\bshoot him\b",
    r"\bshoot her\b",

]


def contains_sensitive_request(message: str) -> bool:

    text = message.lower()

    return any(
        re.search(pattern, text)
        for pattern in SENSITIVE_PATTERNS
    )


def contains_threat(message: str) -> bool:

    text = message.lower()

    return any(
        re.search(pattern, text)
        for pattern in THREAT_PATTERNS
    )


# ============================================================
# STRUCTURED OUTPUT SCHEMA
# ============================================================

RESPONSE_SCHEMA = {

    "type": "object",

    "properties": {

        # IMPORTANT:
        # Action is now at the TOP LEVEL.
        "action": {

            "type": "string",

            "enum": [
                "CONTINUE",
                "CLARIFY",
                "RAG",
                "REFUSE_COMMITMENT",
                "TERMINATE",
                "TERMINATE_SENSITIVE"
            ]
        },

        "understanding": {

            "type": "object",

            "properties": {

                "caller_identity": {
                    "type": ["string", "null"]
                },

                "relationship_to_ram": {
                    "type": ["string", "null"]
                },

                "company": {
                    "type": ["string", "null"]
                },

                "purpose": {
                    "type": ["string", "null"]
                },

                "topic": {
                    "type": ["string", "null"]
                },

                "loan_related": {
                    "type": "boolean"
                },

                "relevance": {

                    "type": "string",

                    "enum": [
                        "RELEVANT",
                        "IRRELEVANT",
                        "INAPPROPRIATE",
                        "UNKNOWN"
                    ]
                },

                "safety": {

                    "type": "string",

                    "enum": [
                        "SAFE",
                        "UNSAFE",
                        "SENSITIVE"
                    ]
                },

                "information": {

                    "type": "object",

                    "properties": {

                        "loan_type": {
                            "type": ["string", "null"]
                        },

                        "loan_amount": {
                            "type": ["string", "null"]
                        },

                        "interest_rate": {
                            "type": ["string", "null"]
                        },

                        "tenure": {
                            "type": ["string", "null"]
                        },

                        "processing_fee": {
                            "type": ["string", "null"]
                        },

                        "other_charges": {
                            "type": ["string", "null"]
                        },

                        "important_conditions": {
                            "type": ["string", "null"]
                        },

                        "callback_information": {
                            "type": ["string", "null"]
                        },

                        "required_action": {
                            "type": ["string", "null"]
                        }
                    },

                    "required": [

                        "loan_type",
                        "loan_amount",
                        "interest_rate",
                        "tenure",
                        "processing_fee",
                        "other_charges",
                        "important_conditions",
                        "callback_information",
                        "required_action"

                    ],

                    "additionalProperties": False
                },

                "missing_information": {

                    "type": "array",

                    "items": {
                        "type": "string"
                    }
                },

                "reason": {

                    "type": "string"
                }
            },

            "required": [

                "caller_identity",
                "relationship_to_ram",
                "company",
                "purpose",
                "topic",
                "loan_related",
                "relevance",
                "safety",
                "information",
                "missing_information",
                "reason"

            ],

            "additionalProperties": False
        },

        "response": {

            "type": "string"
        }
    },

    "required": [

        "action",
        "understanding",
        "response"

    ],

    "additionalProperties": False
}


# ============================================================
# GROQ STRUCTURED CALL
# ============================================================

def call_groq_structured(
    messages: List[Dict[str, str]]
) -> Dict[str, Any]:

    response = client.chat.completions.create(

        model=MODEL,

        messages=messages,

        temperature=0.25,

        max_completion_tokens=1200,

        reasoning_effort="low",

        response_format={

            "type": "json_schema",

            "json_schema": {

                "name": "call_assistant_response",

                "strict": True,

                "schema": RESPONSE_SCHEMA
            }
        }
    )

    text = response.choices[0].message.content

    return json.loads(text)


# ============================================================
# GROQ NORMAL TEXT CALL
# ============================================================

def call_groq_text(
    messages: List[Dict[str, str]]
) -> str:

    response = client.chat.completions.create(

        model=MODEL,

        messages=messages,

        temperature=0.25,

        max_completion_tokens=500,

        reasoning_effort="low"
    )

    return response.choices[0].message.content.strip()


# ============================================================
# MAIN SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """

You are Ram's personal AI call assistant.

IMPORTANT ROLE:

The caller is usually a bank representative, loan agent,
or company representative calling Ram.

You represent RAM.

You do NOT represent the bank.

Your primary job in a loan-related call is:

1. Understand why the caller is calling.
2. Collect the loan/offer details FROM THE CALLER.
3. Identify important missing details.
4. Ask the caller naturally for those missing details.
5. Record information explicitly provided by the caller.
6. Once enough information has been collected, naturally close
   the call and say that you will pass the information to Ram.

------------------------------------------------------------
MOST IMPORTANT RULE
------------------------------------------------------------

The caller is the SOURCE of the loan offer.

Example:

Caller:
"We can offer Ram a personal loan of 10 lakh at 11% for five years."

You should record:

loan_amount = 10 lakh
interest_rate = 11%
tenure = five years

But if the caller has NOT mentioned the processing fee:

DO NOT invent it.

DO NOT calculate it.

DO NOT retrieve a generic processing fee from RAG.

Instead ask the caller naturally:

"Could you also let me know the processing fee for this offer?"

The exact wording must be generated naturally.

------------------------------------------------------------
INFORMATION SOURCE RULE
------------------------------------------------------------

Only put a loan value into information when:

- the caller explicitly provided it
OR
- the caller clearly stated it earlier in the conversation.

Do NOT infer missing offer terms.

Do NOT use general knowledge to fill missing offer details.

Do NOT use the knowledge base to manufacture a bank's offer.

------------------------------------------------------------
RAG RULE
------------------------------------------------------------

RAG is NOT the source of the bank's loan offer.

Use RAG only when the caller genuinely asks for information
that belongs to the assistant's approved knowledge base.

Example:

Caller:
"What documents does Ram need to submit?"

If the knowledge base contains that information,
RAG may be used.

But:

Caller:
"What is the processing fee?"

If the caller is the bank offering the loan and has not
provided the processing fee yet:

DO NOT use RAG.

Ask the caller to provide the processing fee for THEIR offer.

This distinction is extremely important.

------------------------------------------------------------
CONVERSATION MEMORY
------------------------------------------------------------

Use the entire conversation history.

Do not ask again for information already provided.

If the caller gives several details in one sentence,
extract all of them.

------------------------------------------------------------
IMPORTANT LOAN INFORMATION
------------------------------------------------------------

Possible information to collect:

- company
- loan type
- loan amount
- interest rate
- tenure
- processing fee
- other charges
- important conditions
- required documents
- next steps
- callback information

Do not mechanically ask every field.

Use natural conversation.

Ask for the most useful missing information.

------------------------------------------------------------
NATURAL CONVERSATION
------------------------------------------------------------

Do NOT sound like a questionnaire.

Do NOT use fixed predefined sentences.

Generate every response naturally using the LLM.

Ask one or two related missing details at a time.

------------------------------------------------------------
FINANCIAL AUTHORITY
------------------------------------------------------------

You have NO authority to:

- approve a loan
- reject a loan
- sanction a loan
- accept a loan
- confirm that Ram accepts a loan
- negotiate loan terms
- make financial commitments
- claim that Ram agreed to anything

If the caller asks:

"Can you confirm the loan?"

"Can Ram accept this?"

"Can you sanction this?"

"Can you approve this?"

Use:

REFUSE_COMMITMENT

You may summarize what the caller offered.

For example:

"You offered Ram a 10 lakh loan at 11% for five years."

But this does NOT mean Ram accepted it.

------------------------------------------------------------
PASSING INFORMATION TO RAM
------------------------------------------------------------

The assistant can say that it will pass the collected
information to Ram.

For example:

"I've noted the details. I'll pass them to Ram for his review."

This is part of the intended assistant workflow.

------------------------------------------------------------
RELEVANCE
------------------------------------------------------------

RELEVANT:
Related to the purpose of the call.

IRRELEVANT:
Unrelated topic.

INAPPROPRIATE:
Abusive or clearly unsuitable.

UNKNOWN:
Purpose is not yet clear.

------------------------------------------------------------
CLARIFICATION
------------------------------------------------------------

If you do not understand the caller's purpose:

CLARIFY.

If the caller's purpose is clear:

Do not ask unnecessary clarification.

------------------------------------------------------------
ENDING THE CALL
------------------------------------------------------------

When the important information has been collected,
the caller has nothing else to add, or the caller says goodbye:

End naturally.

Do not keep asking unnecessary questions.

------------------------------------------------------------
SAFETY
------------------------------------------------------------

Never handle OTPs, passwords, PINs, MPINs, CVVs,
or card numbers.

If requested, terminate safely.

If the caller becomes threatening or abusive,
terminate politely.

"""


# ============================================================
# UPDATE INFORMATION
# ============================================================

INFORMATION_FIELDS = [

    "loan_type",
    "loan_amount",
    "interest_rate",
    "tenure",
    "processing_fee",
    "other_charges",
    "important_conditions",
    "callback_information",
    "required_action"

]


def update_information(
    state: ConversationState
) -> ConversationState:

    current = dict(
        state["information_collected"]
    )

    extracted = state[
        "understanding"
    ].get(
        "information",
        {}
    )

    # Only update with information actually
    # extracted from the caller.
    for field in INFORMATION_FIELDS:

        value = extracted.get(field)

        if value is not None:

            current[field] = value

    understanding = state["understanding"]

    for field in [

        "company",
        "purpose",
        "topic",
        "caller_identity",
        "relationship_to_ram"

    ]:

        value = understanding.get(field)

        if value is not None:

            current[field] = value

    state["information_collected"] = current

    return state


# ============================================================
# UNDERSTAND + RESPOND
# ============================================================

def understand_and_respond(
    state: ConversationState
):

    latest_message = state[
        "user_message"
    ]

    history = "\n".join(
        state[
            "conversation_history"
        ][-12:]
    )

    previous_information = json.dumps(
        state[
            "information_collected"
        ],
        indent=2,
        ensure_ascii=False
    )

    # --------------------------------------------------------
    # SENSITIVE REQUEST
    # --------------------------------------------------------

    if contains_sensitive_request(
        latest_message
    ):

        response = call_groq_text([

            {
                "role": "system",
                "content": """
You are Ram's personal call assistant.

The caller requested sensitive credentials such as
OTP, PIN, password, CVV, or card number.

Politely explain that you cannot handle such information
and end the call.

Be brief and natural.
"""
            },

            {
                "role": "user",
                "content": latest_message
            }

        ])

        state["understanding"] = {

            "caller_identity": None,
            "relationship_to_ram": None,
            "company": None,
            "purpose": None,
            "topic": None,
            "loan_related": False,

            "relevance": "RELEVANT",

            "safety": "SENSITIVE",

            "information": {

                field: None
                for field in INFORMATION_FIELDS

            },

            "missing_information": [],

            "reason":
                "Sensitive information request."

        }

        state["action"] = (
            "TERMINATE_SENSITIVE"
        )

        state["assistant_response"] = response

        state["call_ended"] = True

        return state

    # --------------------------------------------------------
    # THREAT
    # --------------------------------------------------------

    if contains_threat(
        latest_message
    ):

        response = call_groq_text([

            {
                "role": "system",
                "content": """
You are Ram's personal call assistant.

The caller has made a threatening statement.

Politely end the call.

Be brief and natural.
"""
            },

            {
                "role": "user",
                "content": latest_message
            }

        ])

        state["understanding"] = {

            "caller_identity": None,
            "relationship_to_ram": None,
            "company": None,
            "purpose": None,
            "topic": None,
            "loan_related": False,

            "relevance": "INAPPROPRIATE",

            "safety": "UNSAFE",

            "information": {

                field: None
                for field in INFORMATION_FIELDS

            },

            "missing_information": [],

            "reason":
                "Threatening language detected."

        }

        state["action"] = "TERMINATE"

        state["assistant_response"] = response

        state["call_ended"] = True

        return state

    # --------------------------------------------------------
    # MAIN GROQ REASONING
    # --------------------------------------------------------

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": f"""

PREVIOUS CONVERSATION:

{
    history
    if history
    else "(no previous conversation)"
}


INFORMATION ALREADY COLLECTED:

{previous_information}


LATEST CALLER MESSAGE:

{latest_message}


Analyze the latest caller message in the context
of the entire conversation.

Remember:

The caller is the bank/loan agent.

The caller is the SOURCE of the loan offer.

If an important offer detail is missing,
ASK THE CALLER FOR IT.

Do NOT answer the missing offer detail using RAG.

Use RAG only when the caller genuinely asks
for information belonging to the assistant's
approved knowledge base.

If the caller asks you to approve, accept,
sanction, reject, or confirm a financial decision
on Ram's behalf:

Use REFUSE_COMMITMENT.

Generate a natural response.
"""
        }

    ]

    result = call_groq_structured(
        messages
    )

    state["understanding"] = (
        result["understanding"]
    )

    state["action"] = result["action"]

    state["assistant_response"] = (
        result["response"]
    )

    # Add conversation history
    state[
        "conversation_history"
    ].append(
        f"Caller: {latest_message}"
    )

    state[
        "conversation_history"
    ].append(
        f"Assistant: {result['response']}"
    )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    if result["action"] == "RAG":

        return handle_rag_response(
            state
        )

    # --------------------------------------------------------
    # FINANCIAL COMMITMENT
    # --------------------------------------------------------

    if result["action"] == "REFUSE_COMMITMENT":

        return state

    # --------------------------------------------------------
    # TERMINATION
    # --------------------------------------------------------

    if result["action"] in [

        "TERMINATE",
        "TERMINATE_SENSITIVE"

    ]:

        state["call_ended"] = True

    return state


# ============================================================
# RAG
# ============================================================

def handle_rag_response(
    state: ConversationState
):

    query = state[
        "user_message"
    ]

    try:

        results = retrieve_context(
            query,
            top_k=3
        )

    except Exception:

        results = []

    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

    if not results:

        response = call_groq_text([

            {
                "role": "system",
                "content": """
You are Ram's personal call assistant.

The caller asked for information,
but the approved knowledge base does not
contain the answer.

Do not invent information.

Explain naturally that you do not have
that information.
"""
            },

            {
                "role": "user",
                "content": query
            }

        ])

        state[
            "assistant_response"
        ] = response

        return state

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for item in results:

        if isinstance(item, dict):

            text = item.get(
                "text",
                ""
            )

            source = item.get(
                "source",
                "knowledge base"
            )

        else:

            text = str(item)

            source = "knowledge base"

        context_parts.append(

            f"Source: {source}\n{text}"

        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # GROUNDED ANSWER
    # --------------------------------------------------------

    grounded_response = call_groq_text([

        {
            "role": "system",
            "content": """
You are Ram's personal call assistant.

Answer the caller using ONLY the approved
knowledge-base context below.

Do not invent information.

Do not turn a knowledge-base answer into
a loan offer made by the caller.

Do not approve, reject, accept, or commit
Ram to a financial decision.

Keep the response concise and natural.
"""
        },

        {
            "role": "user",
            "content": f"""

CALLER QUESTION:

{query}


APPROVED KNOWLEDGE BASE:

{context}

"""
        }

    ])

    state[
        "rag_context"
    ] = results

    state[
        "assistant_response"
    ] = grounded_response

    # Replace preliminary response with
    # the final grounded response.
    if state["conversation_history"]:

        state[
            "conversation_history"
        ][-1] = (
            f"Assistant: {grounded_response}"
        )

    return state


# ============================================================
# LANGGRAPH
# ============================================================

def build_graph():

    graph = StateGraph(
        ConversationState
    )

    graph.add_node(
        "understand",
        understand_and_respond
    )

    graph.add_node(
        "information",
        update_information
    )

    graph.set_entry_point(
        "understand"
    )

    graph.add_edge(
        "understand",
        "information"
    )

    graph.add_edge(
        "information",
        END
    )

    return graph.compile()


conversation_graph = build_graph()


# ============================================================
# RUN CALL
# ============================================================

def run_call():

    print("\n" + "=" * 60)

    print(
        "AI CALL ASSISTANT"
    )

    print(
        "GROQ + RAG — LOAN OFFER COLLECTION TEST"
    )

    print("=" * 60)

    state = create_initial_state()

    while not state["call_ended"]:

        try:

            caller_message = input(
                "\nCaller: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\nCall ended by user."
            )

            break

        if not caller_message:

            continue

        state["user_message"] = (
            caller_message
        )

        state["turn_count"] += 1

        state = conversation_graph.invoke(
            state
        )

        print("\nUnderstanding:")

        print(
            json.dumps(
                state["understanding"],
                indent=2,
                ensure_ascii=False
            )
        )

        print(
            f"\nAction: {state['action']}"
        )

        print(
            f"\nAI: {state['assistant_response']}"
        )

        if state["call_ended"]:

            print(
                "\nCall ended."
            )

            break

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL INFORMATION COLLECTED"
    )

    print(
        "=" * 60
    )

    print(
        json.dumps(
            state["information_collected"],
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":

    run_call()