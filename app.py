# Import required libraries
import os
import re
import uvicorn
from typing import List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_cohere import CohereRerank
from fastapi import FastAPI, HTTPException
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define the structure for request body
class QueryRequest(BaseModel):
    query: str

# Define the structure for document details in the response
class DocumentDetails(BaseModel):
    document_name: str
    pages: List[int]

# Define the response structure
class ResponseData(BaseModel):
    answer: str
    relevant_questions: List[str]
    documents: List[DocumentDetails]

class QueryResponse(BaseModel):
    data: ResponseData

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
                loader = PyPDFLoader(file_path)
                pdf_documents = loader.load()
                for doc in pdf_documents:
                    doc.metadata['document_name'] = filename
                documents.extend(pdf_documents)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
        final_documents = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(final_documents, embedding=embeddings)
        vectorstore.save_local(vectorstore_path)
        
        # Create BM25 Retriever
        bm25_retriever = BM25Retriever.from_documents(final_documents)
        
        return vectorstore, bm25_retriever, final_documents
    else:
        print("Loading existing vectorstore...")
        vectorstore = FAISS.load_local(vectorstore_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        
        # We need to recreate the BM25 retriever and documents list
        documents = []
        for filename in os.listdir(pdf_folder):
            if filename.endswith('.pdf'):
                file_path = os.path.join(pdf_folder, filename)
                loader = PyPDFLoader(file_path)
                pdf_documents = loader.load()
                for doc in pdf_documents:
                    doc.metadata['document_name'] = filename
                documents.extend(pdf_documents)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
        final_documents = text_splitter.split_documents(documents)
        
        bm25_retriever = BM25Retriever.from_documents(final_documents)
        
        return vectorstore, bm25_retriever, final_documents

# Attempt to create or load the vectorstore
try:
    vectorstore, bm25_retriever, documents = create_or_load_vectorstore()
except Exception as e:
    print('Not able to load Vectorstore:', e)
    raise

# Create the semantic search retriever
semantic_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Create the hybrid retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.5, 0.5]
)

# Create the CohereRerank compressor for reranking retrieved documents
compressor = CohereRerank(
    model="rerank-english-v2.0",
    top_n=3,
    cohere_api_key=COHERE_API_KEY
)

# Create the ContextualCompressionRetriever with the hybrid retriever and compressor
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever
)

# Initialize the OpenAI language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Define the prompt template for the retrieval QA
decomposition_template = """
Given the following user query, break it down into individual, atomic questions:

User Query: {query}

Please list each atomic question on a new line.
"""
decomposition_prompt = PromptTemplate(template=decomposition_template, input_variables=["query"])

# Create a chain for query decomposition
decomposition_chain = LLMChain(llm=llm, prompt=decomposition_prompt)

# Updated prompt template for answering individual questions
answer_template = """
You are an expert AI assistant tasked with answering the following question based on the given context. Your goal is to provide an accurate, concise, and clear response.

Context: {context}

Question: {question}

Instructions:
1. Provide a concise answer based on the context.
2. If you don't have relevant information in the context, explicitly state that you don't have information on that specific topic.

Format your response as follows:

Answer:
[Your answer here]

Now, please answer the following question:
{question}
"""
answer_prompt = PromptTemplate(template=answer_template, input_variables=["context", "question"])

# Create the RetrievalQA chain with the language model, retriever, and prompt
retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": answer_prompt}
)

# Function to process the query and format the response
def get_answer(query: str) -> Dict:
    """
    Process the query by breaking it down into sub-queries, answering each separately,
    and then combining the results.
    """
    # Decompose the query
    decomposed_queries = decomposition_chain.run(query).strip().split('\n')
    
    combined_answer = ""
    all_relevant_questions = []
    all_documents = []

    # Process each sub-query
    for sub_query in decomposed_queries:
        result = retrievalQA.invoke({"query": sub_query})
        answer = result.get('result', '').strip()
        
        # Extract the answer, skipping any "Answer:" prefix
        answer_match = re.search(r'(?:Answer:)?\s*(.*)', answer, re.DOTALL)
        if answer_match:
            sub_answer = answer_match.group(1).strip()
            combined_answer += f"Question: {sub_query}\n{sub_answer}\n\n"
        
        # Process source documents
        for doc in result.get("source_documents", []):
            doc_info = {
                "document_name": doc.metadata.get('document_name', 'Unknown'),
                "pages": [doc.metadata.get('page', 'Unknown')]
            }
            if doc_info not in all_documents:
                all_documents.append(doc_info)

    # Generate a summary and relevant questions based on the combined answer
    summary_prompt = PromptTemplate(
        template="Summarize the following information and provide 3 relevant follow-up questions:\n\n{combined_answer}",
        input_variables=["combined_answer"]
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    summary_result = summary_chain.run(combined_answer).strip()

    # Extract summary and relevant questions
    summary_match = re.search(r'Summary:(.*?)(?:Follow-up Questions:|$)', summary_result, re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip()
    else:
        summary = "Unable to generate summary."

    relevant_questions_match = re.search(r'Follow-up Questions:(.*?)$', summary_result, re.DOTALL)
    if relevant_questions_match:
        relevant_questions = relevant_questions_match.group(1).strip().split('\n')
        all_relevant_questions = [q.strip() for q in relevant_questions if q.strip()]
    else:
        all_relevant_questions = ["No relevant questions found."]

    response_data = {
        "data": {
            "answer": summary,
            "relevant_questions": all_relevant_questions,
            "documents": all_documents
        }
    }

    return response_data

# FastAPI route to handle the query
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=QueryResponse)
async def ask(query_request: QueryRequest):
    try:
        result = get_answer(query_request.query)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)