import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables from .env file (for your GOOGLE_API_KEY)
load_dotenv()

# Define paths
DATA_PATH = "documents/"
CHROMA_PATH = "chroma_db"


def load_and_split_documents():
    print("Loading documents...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    if not documents:
        print("No documents found.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    print("Splitting documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")
    return chunks


def create_and_store_embeddings(chunks):
    print("Creating and storing embeddings...")
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(chunks, embedding_model, persist_directory=CHROMA_PATH)
    print(f"Successfully created and stored embeddings in '{CHROMA_PATH}'.")
    return db


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

if __name__ == "__main__":
    # Check if the database needs to be created
    if not os.path.exists(CHROMA_PATH):
        document_chunks = load_and_split_documents()
        if document_chunks:
            create_and_store_embeddings(document_chunks)

    # Load the existing database with Google embeddings
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    retriever = db.as_retriever(search_kwargs={"k": 2})

    # Use a model name confirmed to be available to your API key
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    while True:
        query = input("\nEnter a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        result = chain.invoke(query)
        print("\n--- Answer ---")
        print(result)
        print("----------------\n")