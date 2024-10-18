import os
import uvicorn
from typing import List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_cohere import CohereRerank
from fastapi import FastAPI, HTTPException
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Define folder for PDFs and path for the vectorstore
pdf_folder = "data/"
vectorstore_path = os.path.join(pdf_folder, 'vectorstore.faiss')

# Ensure the PDF folder exists
os.makedirs(pdf_folder, exist_ok=True)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

def create_or_load_vectorstore():
    """
    Create or load a vectorstore from PDF documents in the specified folder.
    If no PDFs are present, it creates a new vectorstore from them.
    """
    if not os.listdir(pdf_folder) or not os.path.exists(vectorstore_path):
        print("Creating new vectorstore from PDFs...")
        documents = []
        for filename in os.listdir(pdf_folder):
            if filename.endswith('.pdf'):
                file_path = os.path.join(pdf_folder, filename)
                loader = PyPDFLoader(file_path)  # Load PDF documents
                pdf_documents = loader.load()
                for doc in pdf_documents:
                    doc.metadata['document_name'] = filename  # Add document name metadata
                documents.extend(pdf_documents)
        
        # Split documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
        final_documents = text_splitter.split_documents(documents)

        # Create a FAISS vectorstore from the documents
        vectorstore = FAISS.from_documents(final_documents, embedding=embeddings)
        vectorstore.save_local(vectorstore_path)  # Save the vectorstore locally
    else:
        print("Loading existing vectorstore...")
        vectorstore = FAISS.load_local(vectorstore_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# Attempt to create or load the vectorstore
try:
    vectorstore = create_or_load_vectorstore()
except Exception as e:
    print('Not able to load Vectorstore:', e)

# Create the base retriever for similarity search
base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Create the CohereRerank compressor for reranking retrieved documents
compressor = CohereRerank(
    model="rerank-english-v2.0",
    top_n=3,
    cohere_api_key=COHERE_API_KEY
)

# Create the ContextualCompressionRetriever with the base retriever and compressor
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Initialize the OpenAI language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Define the prompt template for the retrieval QA
prompt_template = """You are tasked with answering the user's question based on the given context. Your answer must be concise, clear, and based only on the provided information.
Context:
{context}
Question:
{question}
Answer:"""

# Create the prompt using the defined template
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create the RetrievalQA chain with the language model, retriever, and prompt
retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

def ask_question(query: str):
    """
    Ask a question using the RetrievalQA chain and return the answer along with the source documents.
    """
    result = retrievalQA.invoke({"query": query})
    return result['result'], result['source_documents']

# Example usage
query = "What is monkey pox?"
answer, source_docs = ask_question(query)

# Print the question and answer
print("Question:", query)
print('\n')
print("Answer:", answer)

# Print the source documents that contributed to the answer
print("\nSource Documents:")
for i, doc in enumerate(source_docs, 1):
    print(f"\nDocument {i}:")
    print(f"Content: {doc.page_content[:150]}...")  # Show the first 150 characters
    print(f"Source: {doc.metadata.get('document_name', 'Unknown')}")  # Print document name or 'Unknown'