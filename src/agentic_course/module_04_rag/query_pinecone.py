import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"
# Must match the dimension of the Pinecone index you created (384 for all-MiniLM-L6-v2)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def answer_from_index(
    question: str,
    index_name: str,
    embeddings: Embeddings,
    k: int = 3,
) -> str:
    """Retrieve top-k chunks from a Pinecone index and generate a grounded answer."""
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    docs = retriever.invoke(question)
    context = "\n\n".join(f"- {doc.page_content}" for doc in docs)

    llm = init_chat_model(f"groq:{GROQ_MODEL}", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer using ONLY the retrieved context. If the answer is missing, say so.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return str(response.content)


if __name__ == "__main__":
    index_name = os.environ["INDEX_NAME"]
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    output = answer_from_index(
        question="How does LangGraph manage state?",
        index_name=index_name,
        embeddings=embeddings,
        k=3,
    )
    print(output)