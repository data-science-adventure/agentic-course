import os
import time

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

EMBEDDING_DIMENSION = 384  # Matches sentence-transformers/all-MiniLM-L6-v2


def ensure_pinecone_index(
    pc: Pinecone,
    index_name: str,
    dimension: int = 384,
    metric: str = "cosine",
    reset: bool = False,
) -> None:
    """Create a dense serverless Pinecone index, optionally resetting an existing one."""
    if pc.has_index(index_name):
        if reset:
            print(f"Index '{index_name}' found; deleting because RESET_INDEX=true...")
            pc.delete_index(index_name)
            while pc.has_index(index_name):
                time.sleep(1)
        else:
            print(f"Index '{index_name}' already exists; keeping existing index.")
            return

    print(f"Creating index '{index_name}' (dimension={dimension}, metric={metric})...")
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    while not pc.describe_index(index_name).status.get("ready", False):
        time.sleep(1)

    print(f"Index '{index_name}' created and ready.")


def main():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["INDEX_NAME"]
    reset_index = os.getenv("RESET_INDEX", "false").strip().lower() == "true"

    ensure_pinecone_index(
        pc=pc,
        index_name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        reset=reset_index,
    )


if __name__ == "__main__":
    main()
