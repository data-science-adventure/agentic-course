import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions
GROQ_MODEL = "openai/gpt-oss-20b"


def get_vector_store(index_name: str, namespace: str | None = None) -> PineconeVectorStore:
    """Connect to an existing Pinecone index and bind HuggingFaceEmbeddings."""
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)

    stats = index.describe_index_stats()
    print(f"[Pinecone Stats for '{index_name}']: {stats}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace,
    )


def seed_docs(vector_store: PineconeVectorStore) -> None:
    """Upsert sample course documents into the Pinecone index."""
    docs = [
        Document(
            page_content=(
                "LangChain provides model wrappers, tool calling, prompts, "
                "and retrievers as composable components."
            ),
            metadata={"source": "course_notes", "module": "module_04_rag"},
        ),
        Document(
            page_content=(
                "LangGraph models agent workflows as a StateGraph with typed state "
                "(@TypedDict / Pydantic), conditional edges, and checkpointers."
            ),
            metadata={"source": "course_notes", "module": "module_03_langgraph"},
        ),
        Document(
            page_content=(
                "A Pinecone index stores dense vectors and metadata, and must match "
                "the embedding model's vector dimension (e.g., 384 for all-MiniLM-L6-v2)."
            ),
            metadata={"source": "course_notes", "module": "module_04_rag"},
        ),
    ]
    ids = vector_store.add_documents(docs)
    print(f"[Seeded] Upserted {len(ids)} documents: {ids}")


def ask_rag(vector_store: PineconeVectorStore, question: str, k: int = 3) -> str:
    """Retrieve relevant chunks and generate a grounded response via ChatGroq."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(question)
    print(f"[Retrieved {len(retrieved_docs)} doc(s) for query: {question!r}]")

    if not retrieved_docs:
        return "I’m sorry, but I don’t have any context to answer that question."

    context = "\n\n".join(f"- {doc.page_content}" for doc in retrieved_docs)
    llm = init_chat_model(f"groq:{GROQ_MODEL}", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer using ONLY the provided context. If the context does not contain "
                "the answer, state that clearly.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    response = (prompt | llm).invoke({"context": context, "question": question})
    return str(response.content)


if __name__ == "__main__":
    index_name = os.environ["INDEX_NAME"]
    namespace = os.getenv("PINECONE_NAMESPACE") or None

    store = get_vector_store(index_name=index_name, namespace=namespace)

    # Seed once (or whenever SEED_DOCS=true is set in environment)
    if os.getenv("SEED_DOCS", "false").strip().lower() == "true":
        seed_docs(store)

    answer = ask_rag(store, "How does LangGraph manage state?", k=3)
    print("\nFinal Answer:\n" + answer)