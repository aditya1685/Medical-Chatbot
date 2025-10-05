import os
import sys
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda
load_dotenv()
from langchain_groq import ChatGroq

FAISS_INDEX_PATH = r'D:\Medical Chatbot\Medical-Chatbot\data\faiss_index_medical_book'
PDF_FILE_PATH = r'D:\Medical Chatbot\Medical-Chatbot\data\Medical_book.pdf'

model = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0.0
)

embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-large-en-v1.5")

if not os.path.exists(FAISS_INDEX_PATH):
    print("Creating new FAISS index and saving to disk...")
    loader = PyPDFLoader(PDF_FILE_PATH)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 150
    )

    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    # Save the index and associated documents to the path
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("FAISS index successfully created and saved for future use.")
else:
    print("Loading existing FAISS index from disk...")
    # Load the index from the path
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded successfully.")

# --- RAG Chain Setup ---
retriever = vectorstore.as_retriever(search_type = 'similarity', kwargs = {"k": 4})

parser = StrOutputParser()

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

prompt = PromptTemplate(
    template="""
       ## ROLE: Hybrid AI Assistant
      
      You are an intelligent assistant. Your goal is to answer the user's question.
      
      1. **PRIORITY 1 (Medical):** If the <CONTEXT> contains relevant medical information, use it to provide a highly accurate, grounded answer.
      2. **PRIORITY 2 (General):** If the <CONTEXT> clearly states that no relevant information was found (or if the question is general knowledge, like geography or history), answer the question using your general knowledge base.
      just return the answer, don't mention about the availability of the context.
      <CONTEXT>
      {context}
      </CONTEXT>
      
      Question: {question}
      
      ANSWER:
    """,
    input_variables = ['context', 'question']
)

chain1 = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

chain2 = chain1 | prompt | model | parser

# --- Execution ---
result = chain2.invoke('What is the capital of india?')
print("--- Final Answer ---")
print(result)
