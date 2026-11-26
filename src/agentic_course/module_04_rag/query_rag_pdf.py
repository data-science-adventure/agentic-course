import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"


def query_rag_document(
    question: str,
    index_name: str | None = None,
    namespace: str | None = None,
    k: int = 4,
) -> str:
    """Retrieve relevant PDF chunks from Pinecone and answer via ChatGroq."""
    target_index = index_name or os.environ["INDEX_NAME"]
    target_namespace = (
        namespace
        if namespace is not None
        else (os.getenv("PINECONE_NAMESPACE") or None)
    )

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(target_index)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=target_namespace,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    if not docs:
        return "No matching passages found in the Pinecone index."

    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 'N/A')}] {doc.page_content}" for doc in docs
    )

    llm = init_chat_model(f"groq:{GROQ_MODEL}", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's question using ONLY the provided PDF context. "
                "Include page references when available. If the context does not contain "
                "the answer, state that clearly.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    response = (prompt | llm).invoke({"context": context, "question": question})
    return str(response.content)


def main():
    sample_question = (
        "What is the vocabulary mismatch problem in sparse retrieval, and how does "
        "dense retrieval or a bi-encoder help address it?"
    )
    print(f"Question: {sample_question}\n")
    answer = query_rag_document(question=sample_question, k=4)
    print(f"Answer:\n{answer}")


if __name__ == "__main__":
    main()
