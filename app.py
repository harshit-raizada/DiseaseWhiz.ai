# Import required libraries
import os
import uvicorn
from typing import List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_cohere import CohereRerank
from fastapi import FastAPI, HTTPException
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
prompt_template = """
You are tasked with answering the user's question based on the given context, and provide a list of three relevant follow-up questions. Your answer must be concise, clear, and based only on the provided information.
Context: {context}
Answer:
[Provide the answer]
Relevant questions:
[List three relevant questions]
Question: {question}
"""

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

# Function to process the query and format the response
def get_answer(query: str) -> Dict:
    result = retrievalQA.invoke({"query": query})
    split_output = result.get('result', '').split('Answer:')
    answer = split_output[-1].strip() if len(split_output) > 1 else split_output[0].strip()
    response_data = {
        "data": {
            "answer": "",
            "relevant_questions": [],
            "documents": []
        }
    }
    if "The provided context does not contain information about" in answer:
        response_data["data"]["answer"] = "Sorry! The model does not know the answer to this question."
    else:
        markers = ["Relevant questions:", "Relevant questions related to the user's query:", "relevant questions",
                   "Recommend questions", "recommend questions", "Here are some relevant questions:",
                   "Here are some related questions:", "Other questions you might find useful:", "Other relevant questions:"]
        response_parts = None
        for marker in markers:
            if marker in answer:
                response_parts = answer.split(marker)
                break
        if response_parts:
            ans, reco_ques = response_parts[0].strip(), response_parts[1].strip()
            response_data["data"]["answer"] = ans
            response_data["data"]["relevant_questions"] = [q.strip() for q in reco_ques.split('\n') if q.strip()]
        else:
            response_data["data"]["answer"] = answer
            response_data["data"]["relevant_questions"] = ["No relevant questions provided."]
    document_pages = {}
    for doc in result.get("source_documents", []):
        document_name = doc.metadata.get('document_name', 'Unknown')
        page_number = doc.metadata.get('page', 'Unknown')
        if document_name not in document_pages:
            document_pages[document_name] = {"pages": set()}
        if page_number != 'Unknown':
            document_pages[document_name]["pages"].add(page_number)
    for document_name, doc_data in document_pages.items():
        response_data["data"]["documents"].append({
            "document_name": document_name,
            "pages": sorted(doc_data["pages"])
        })
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