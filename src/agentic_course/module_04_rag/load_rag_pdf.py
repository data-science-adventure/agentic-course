import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

# Must match the Pinecone index dimension (384 for sentence-transformers/all-MiniLM-L6-v2)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PDF_PATH = "rag.pdf"
BATCH_SIZE = 100


def load_and_split_pdf(
    pdf_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    """Load every page of the PDF and split into overlapping text chunks."""
    resolved_path = Path(pdf_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"PDF not found at: {resolved_path}")

    loader = PyPDFLoader(str(resolved_path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    print(
        f"Loaded {len(pages)} page(s) from '{resolved_path}' "
        f"and split into {len(chunks)} chunk(s)."
    )
    return chunks


def ingest_pdf_to_pinecone(
    pdf_path: str | Path = DEFAULT_PDF_PATH,
    index_name: str | None = None,
    namespace: str | None = None,
) -> None:
    """Upsert all split PDF chunks into the target Pinecone index."""
    target_index = index_name or os.environ["INDEX_NAME"]
    target_namespace = (
        namespace
        if namespace is not None
        else (os.getenv("PINECONE_NAMESPACE") or None)
    )

    chunks = load_and_split_pdf(pdf_path)

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(target_index)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=target_namespace,
    )

    total_upserted = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        ids = vector_store.add_documents(batch)
        total_upserted += len(ids)

    stats = index.describe_index_stats()
    print(
        f"Upserted {total_upserted} chunk(s) into index '{target_index}' "
        f"(namespace={target_namespace!r}). Total index vectors: {stats.get('total_vector_count')}."
    )


def main():
    pdf_file = os.getenv("RAG_PDF_PATH", DEFAULT_PDF_PATH)
    ingest_pdf_to_pinecone(pdf_path=pdf_file)


if __name__ == "__main__":
    main()
