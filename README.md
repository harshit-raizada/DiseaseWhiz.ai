### DiseaseWhiz.ai

DiseaseWhiz.ai is an AI-powered chatbot designed to provide accurate and comprehensive information about various diseases, including their history, symptoms, and effective management strategies. With a user-friendly interface, DiseaseWhiz.ai aims to educate and empower users by offering reliable insights into health topics such as mental disorders, STIs, dengue, herpes, monkeypox, and Zika.

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

- `git clone https://github.com/your-repo/DiseaseWhiz.ai.git`

- `cd DiseaseWhiz.ai`

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

`
{
    "query": "What is Cancer and is it curable?"
}
`

### Response

```
{
    "data": {
        "answer": "**\nCancer is a broad term for diseases characterized by the rapid growth of abnormal cells that can invade nearby tissues and spread to other organs (metastasis). There are various types of cancer, including breast, lung, colon, prostate, skin, and stomach cancers, with cervical cancer being notably prevalent in 23 countries. The development of cancer involves the transformation of normal cells into tumor cells, influenced by genetic factors and external agents such as physical, chemical, and biological carcinogens. Diagnosis typically involves early detection through awareness of symptoms and screening methods. Treatment options vary based on cancer type and may include surgery, radiation, and medications. Many cancers can be cured if detected early, but the likelihood of a cure depends on factors like cancer type, stage at diagnosis, treatment availability, and socioeconomic conditions.\n\n**",
        "relevant_questions": [
            "**",
            "1. What are the most effective screening methods for early detection of different types of cancer?",
            "2. How do socioeconomic factors impact access to cancer treatment and outcomes in different regions?",
            "3. What advancements are being made in cancer research to improve treatment options and cure rates?"
        ],
        "documents": [
            {
                "document_name": "Cancer.pdf",
                "pages": [
                    0
                ]
            },
            {
                "document_name": "Cancer.pdf",
                "pages": [
                    1
                ]
            },
            {
                "document_name": "Cancer.pdf",
                "pages": [
                    3
                ]
            },
            {
                "document_name": "Cancer.pdf",
                "pages": [
                    7
                ]
            },
            {
                "document_name": "Cancer.pdf",
                "pages": [
                    8
                ]
            },
            {
                "document_name": "Cancer.pdf",
                "pages": [
                    4
                ]
            },
            {
                "document_name": "Cancer.pdf",
                "pages": [
                    9
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

If you encounter any issues or have questions, feel free to open an issue on GitHub or reach out on [LinkedIn](https://www.linkedin.com/in/harshit-raizada/)
