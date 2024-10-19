### DiseaseWhiz.ai

Welcome to the Intelligent Document Query System! This system allows users to ask natural language questions based on a collection of PDF documents, retrieving the most relevant answers and providing follow-up questions. It combines powerful retrievers and large language models to generate responses and summarize information efficiently.

## Features

- Hybrid Search: Combines semantic search (using FAISS and OpenAI embeddings) with keyword search (BM25 retriever) for highly relevant document retrieval.
- Contextual Compression: Utilizes Cohere’s powerful re-ranking model to prioritize the most important document snippets.
- LLM-Powered QA: Answers user queries with the assistance of GPT-4o-mini model.
- PDF Document Loader: Automatically loads, splits, and indexes PDFs for retrieval.
- Query Decomposition: Decomposes complex queries into atomic questions for granular answers.
- Follow-Up Suggestions: Provides relevant follow-up questions after answering the initial query.

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.8 or above
- An OpenAI API key
- A Cohere API key
- Uvicorn for running the FastAPI server
- Installation & Setup

### Clone the Repository

- `git clone https://github.com/your-repo/intelligent-document-query-system.git`

- `cd intelligent-document-query-system`

### Set Up a Virtual Environment
   
We recommend using a virtual environment to manage dependencies.

- `python -m venv venv`

- `venv\Scripts\activate`

### Install the Required Dependencies

- `pip install -r requirements.txt`

### Configure Environment Variables
   
You’ll need to add your API keys by creating a .env file in the root of the project:

Inside the .env file, add the following lines:

- `OPENAI_API_KEY=your_openai_api_key`

- `COHERE_API_KEY=your_cohere_api_key`

### Add Your PDF Documents
   
Place any PDF documents you want to use for question-answering in the data/ folder within the project directory. These will be automatically indexed and used for retrieval.

### Run the Application
   
You can start the FastAPI server by running:

- `uvicorn main:app --reload` or `python app.py`

The API will be accessible at http://localhost:8000.

## Usage

### Health Check

To ensure the service is running, you can perform a health check by navigating to:

`GET /health`
You should receive a response:

`{
  "status": "healthy"
}
`

### Asking Questions

To query the system, use the /ask endpoint. Send a POST request with a query in the following format:

### Request

`POST /ask`

`{
  "query": "What is Dengue?"
}
`

### Response

```{
    "data": {
        "answer": "**\nZika is a viral infection primarily transmitted by Aedes mosquitoes, particularly Aedes aegypti. It often presents mild symptoms such as fever, rash, and conjunctivitis, but can lead to severe complications, especially during pregnancy, including microcephaly in infants and Guillain-Barré syndrome. Zika can also be transmitted through sexual contact, blood transfusions, and organ transplants. There is currently no vaccine or specific treatment for Zika. Prevention strategies include avoiding travel to affected areas, vector control, community clean-up campaigns, and personal protective measures. Zika is commonly found in tropical and subtropical regions, with significant risks for pregnant women.\n\n**",
        "relevant_questions": [
            "**",
            "1. What are the current research efforts focused on developing a vaccine or treatment for Zika?",
            "2. How can communities effectively implement vector control strategies to reduce the risk of Zika transmission?",
            "3. What specific guidelines should pregnant women follow if they are in or traveling to areas with reported Zika cases?"
        ],
        "documents": [
            {
                "document_name": "9789241565530-eng.pdf",
                "pages": [
                    95
                ]
            },
            {
                "document_name": "9789241565530-eng.pdf",
                "pages": [
                    97
                ]
            },
            {
                "document_name": "9789241565530-eng.pdf",
                "pages": [
                    103
                ]
            },
            {
                "document_name": "9789241565530-eng.pdf",
                "pages": [
                    99
                ]
            },
            {
                "document_name": "9789241565530-eng.pdf",
                "pages": [
                    110
                ]
            },
            {
                "document_name": "9789241565530-eng.pdf",
                "pages": [
                    98
                ]
            },
            {
                "document_name": "9789241565530-eng.pdf",
                "pages": [
                    101
                ]
            },
            {
                "document_name": "9789241565530-eng.pdf",
                "pages": [
                    100
                ]
            },
            {
                "document_name": "9789241565530-eng.pdf",
                "pages": [
                    102
                ]
            },
            {
                "document_name": "9789241565530-eng.pdf",
                "pages": [
                    96
                ]
            }
        ]
    }
}
```

## Project Structure

- **main.py**: The core FastAPI application with the query handler logic.
- **data/**: A folder where all PDF documents are stored and indexed.
- **vectorstore.faiss**: The FAISS index file for semantic document retrieval.
- **requirements.txt**: List of Python dependencies required for the project.
- **.env**: File containing your API keys.

## Technologies Used

- **FastAPI**: A modern, fast web framework for building APIs with Python.
- **LangChain**: Chain-of-thought framework for language models.
- **FAISS**: Facebook AI Similarity Search for semantic document retrieval.
- **BM25**: Term-based retriever for traditional keyword search.
- **OpenAI**: GPT models for natural language processing.
- **Cohere**: For document re-ranking and contextual compression.
- **Uvicorn**: Lightning-fast ASGI server for FastAPI.

## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository and submit a pull request.

Fork the repository.

- Create a new branch: `git checkout -b feature-branch-name`.
- Make your changes.
- Commit your changes: `git commit -m 'Add some feature'`.
- Push to the branch: `git push origin feature-branch-name`.
- Submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Support

If you encounter any issues or have questions, feel free to open an issue on GitHub or reach out on [https://www.linkedin.com/in/harshit-raizada/]
