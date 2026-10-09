import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
)

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY or GOOGLE_API_KEY not found in .env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-001"

KNOWLEDGE_BASE_DIR = Path("knowledge_base")

CHROMA_DB_DIR = "chroma_db"

COLLECTION_NAME = "call_assistant_kb"

MAX_CHUNK_CHARS = 300

# Maximum acceptable ChromaDB cosine distance.
#
# Lower distance = more relevant.
#
# Based on the current knowledge base tests:
#
# Relevant queries:
#   ~0.27 - 0.35
#
# Irrelevant query:
#   ~0.48+
#
# Therefore 0.40 gives us a reasonable safety boundary.
MAX_DISTANCE = 0.40


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "hnsw:space": "cosine"
    }
)


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

def load_knowledge_base():

    documents = []

    if not KNOWLEDGE_BASE_DIR.exists():

        print(
            f"Knowledge base directory not found: "
            f"{KNOWLEDGE_BASE_DIR}"
        )

        return documents

    for file_path in KNOWLEDGE_BASE_DIR.glob("*.txt"):

        try:

            text = file_path.read_text(
                encoding="utf-8"
            )

        except Exception as error:

            print(
                f"Could not read "
                f"{file_path.name}: {error}"
            )

            continue

        text = text.strip()

        if not text:
            continue

        documents.append(
            {
                "source": file_path.name,
                "text": text
            }
        )

    return documents


# ============================================================
# CHUNK TEXT
# ============================================================

def chunk_text(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS
):

    chunks = []

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    current_chunk = ""

    for paragraph in paragraphs:

        if not current_chunk:

            current_chunk = paragraph

        elif (
            len(current_chunk)
            + len(paragraph)
            + 1
            <= max_chars
        ):

            current_chunk += (
                "\n" + paragraph
            )

        else:

            chunks.append(
                current_chunk
            )

            current_chunk = paragraph

    if current_chunk:

        chunks.append(
            current_chunk
        )

    return chunks


# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks():

    source_documents = (
        load_knowledge_base()
    )

    chunks = []

    for document in source_documents:

        text_chunks = chunk_text(
            document["text"]
        )

        for index, chunk in enumerate(
            text_chunks
        ):

            chunks.append(
                {
                    "id":
                        f"{document['source']}_{index}",

                    "source":
                        document["source"],

                    "text":
                        chunk
                }
            )

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    texts,
    task_type
):

    if not texts:

        return []

    response = client.models.embed_content(

        model=EMBEDDING_MODEL,

        contents=texts,

        config={
            "task_type": task_type
        }
    )

    return [
        embedding.values
        for embedding in response.embeddings
    ]


# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

def build_knowledge_base():

    chunks = create_chunks()

    if not chunks:

        print(
            "No knowledge-base chunks found."
        )

        return False

    print(
        f"Creating embeddings for "
        f"{len(chunks)} chunks..."
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = create_embeddings(
        texts,
        "RETRIEVAL_DOCUMENT"
    )

    if len(embeddings) != len(chunks):

        print(
            "Embedding count does not match "
            "knowledge-base chunk count."
        )

        return False

    # --------------------------------------------------------
    # Clear old collection data
    # --------------------------------------------------------

    existing = collection.get()

    existing_ids = existing.get(
        "ids",
        []
    )

    if existing_ids:

        collection.delete(
            ids=existing_ids
        )

    # --------------------------------------------------------
    # Add new knowledge-base data
    # --------------------------------------------------------

    collection.add(

        ids=[
            chunk["id"]
            for chunk in chunks
        ],

        documents=[
            chunk["text"]
            for chunk in chunks
        ],

        embeddings=embeddings,

        metadatas=[
            {
                "source": chunk["source"]
            }
            for chunk in chunks
        ]
    )

    print(
        f"Knowledge base built successfully "
        f"with {len(chunks)} chunks."
    )

    return True


# ============================================================
# ENSURE KNOWLEDGE BASE
# ============================================================

def ensure_knowledge_base():

    count = collection.count()

    if count == 0:

        print(
            "Knowledge base is empty. "
            "Building knowledge base..."
        )

        return build_knowledge_base()

    print(
        f"Knowledge base already exists "
        f"with {count} chunks."
    )

    return True


# ============================================================
# RETRIEVE CONTEXT
# ============================================================

def retrieve_context(
    query: str,
    top_k: int = 3
):

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query:

        return []

    query = query.strip()

    if not query:

        return []

    # --------------------------------------------------------
    # Make sure knowledge base exists
    # --------------------------------------------------------

    if not ensure_knowledge_base():

        return []

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embeddings = create_embeddings(
        [query],
        "RETRIEVAL_QUERY"
    )

    if not query_embeddings:

        return []

    query_embedding = query_embeddings[0]

    # --------------------------------------------------------
    # Query ChromaDB
    # --------------------------------------------------------

    results = collection.query(

        query_embeddings=[
            query_embedding
        ],

        n_results=top_k,

        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    retrieved = []

    # --------------------------------------------------------
    # Apply relevance threshold
    # --------------------------------------------------------

    for index, document in enumerate(
        documents
    ):

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        # No valid distance means we cannot
        # safely determine relevance.
        if distance is None:

            continue

        # Reject weak matches.
        if distance > MAX_DISTANCE:

            continue

        retrieved.append(
            {
                "text": document,

                "source":
                    metadata.get(
                        "source",
                        "unknown"
                    ),

                "distance": distance
            }
        )

    return retrieved


# ============================================================
# TEST RETRIEVAL
# ============================================================

def run_tests():

    if not ensure_knowledge_base():

        print(
            "Unable to initialize "
            "knowledge base."
        )

        return

    test_queries = [

        "What is the maximum personal loan amount?",

        "What is the processing fee?",

        "What is the loan tenure?",

        "What is the interest rate?",

        "Who won the cricket world cup?"
    ]

    print("\n" + "=" * 60)

    print(
        "RAG RETRIEVAL TEST"
    )

    print("=" * 60)

    print(
        f"\nMaximum accepted distance: "
        f"{MAX_DISTANCE}"
    )

    for query in test_queries:

        print(
            f"\nQuery: {query}"
        )

        results = retrieve_context(
            query,
            top_k=3
        )

        if not results:

            print(
                "No relevant context found."
            )

            continue

        for index, result in enumerate(
            results,
            1
        ):

            print(
                f"\nResult {index}"
            )

            print(
                f"Source: "
                f"{result['source']}"
            )

            print(
                f"Distance: "
                f"{result['distance']}"
            )

            print(
                f"Text: "
                f"{result['text']}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_tests()