import json
import requests
import streamlit as st
from typing import List, Dict

# Define the URL of your FastAPI backend
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="🩺 DiseaseWhiz.ai")

# Custom CSS to make the UI more attractive and formal
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;  /* Light gray background */
        padding: 2rem;  /* Add padding for better spacing */
    }
    .stButton>button {
        background-color: #007bff;  /* Primary blue button */
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stTextInput>div>div>input {
        background-color: #fff;
        color: #333;
        border-radius: 4px;
        border: 1px solid #ccc;  /* Add border for definition */
        text-align: center;  /* Center text input */
    }
    .stHeader {
        background-color: #343a40;  /* Dark gray header */
        padding: 1rem;
        color: white;  /* White font color for better visibility */
        text-align: center;  /* Center the text */
        font-size: 2.5rem;  /* Increase font size for header */
        border-radius: 8px;  /* Rounded corners for header */
    }
    .stSubheader {
        color: #007bff;  /* Blue for subheaders for visibility */
        font-size: 1.5rem;  /* Increase font size for subheaders */
    }
    .answer-box {
        background-color: #e9ecef;  /* Light gray background for answer box */
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);  /* More subtle shadow */
        margin-top: 1.5rem;
        font-size: 1.25rem;  /* Larger font size for answers */
        line-height: 1.6;  /* Increase line height for readability */
        color: #333;  /* Dark text color for visibility */
        border-left: 5px solid #007bff;  /* Add a left border for accent */
    }
    .question-box {
        margin-top: 1rem;
        padding: 0.5rem 1rem;
        background-color: #f8f9fa;  /* Light background for questions */
        border-radius: 4px;
        font-size: 1.1rem;
        color: #333;
    }
    .document-box {
        background-color: #f8f9fa;  /* Light gray for document box */
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        border-left: 4px solid #28a745;  /* Green accent border */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
        color: #333;  /* Dark text color for visibility */
    }
    .unordered-list {
        list-style-type: disc;  /* Bullet list */
        padding-left: 1.5rem;
        margin: 0;
    }
    .sidebar {
        background-color: #f1f3f5;  /* Light background for sidebar */
        padding: 1rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="stHeader">🩺 DiseaseWhiz.ai 💬</h1>', unsafe_allow_html=True)

    # Sidebar for descriptions
    st.sidebar.title("About")
    st.sidebar.markdown("""
        This application features an AI-powered chatbot designed to answer your queries about various diseases 
        listed by the World Health Organization (WHO). The chatbot leverages advanced RAG 
        techniques to provide accurate and reliable information. 

        **Key Features:**
        - Get detailed explanations about diseases, including symptoms, prevention, and treatment options.
        - Access information on mental disorders, STIs, dengue, herpes, monkeypox, Zika, and more.
        - Receive answers based on the latest WHO guidelines and recommendations.
    """)

    # Centered user input
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    query = st.text_input("Enter your question:", placeholder="Type your question here...")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Get Answer"):
        if query:
            try:
                # Send request to backend
                response = requests.post(f"{BACKEND_URL}/ask", json={"query": query})
                response.raise_for_status()
                result = response.json()

                # Display answer
                st.markdown('<h4 class="stSubheader">Answer:</h4>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box">{result["data"]["answer"]}</div>', unsafe_allow_html=True)

                # Display relevant questions
                st.markdown('<h4 class="stSubheader">Relevant Questions:</h4>', unsafe_allow_html=True)
                st.markdown('<ul class="unordered-list">', unsafe_allow_html=True)
                for question in result["data"]["relevant_questions"]:
                    st.markdown(f"<li>{question}</li>", unsafe_allow_html=True)
                st.markdown('</ul>', unsafe_allow_html=True)

                # Display document sources
                st.markdown('<h4 class="stSubheader">Source Documents:</h4>', unsafe_allow_html=True)
                for doc in result["data"]["documents"]:
                    st.markdown(f'<div class="document-box"><strong>Document:</strong> {doc["document_name"]}<br><strong>Pages:</strong> {", ".join(map(str, doc["pages"]))}</div>', unsafe_allow_html=True)

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()