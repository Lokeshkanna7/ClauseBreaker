# Student Lease Break Advisor

### A RAG-powered AI assistant that cross-references lease agreements with state property laws to provide grounded legal information.

Overview

This project was built to solve a specific problem: Students often do not understand their rights when trying to break a lease. 

Generic LLMs hallucinate legal advice. To solve this, I built a **Retrieval-Augmented Generation (RAG)** pipeline that grounds answers in two specific sources of truth:
1.  **The User's Specific Contract** (PDF/DOCX Upload)
2.  **Verifiable State Statutes** (Real Property Law)

Note: This project was built as a rapid prototype to demonstrate Applied AI capabilities in the Legal Tech domain.*

Key Features

* **Dual-Context Retrieval:** simultaneously queries the specific lease agreement AND relevant state laws to form a comprehensive answer.
* **Document Parsing:** Handles `.pdf` and `.docx` files using LangChain loaders.
* **Intelligent Chunking:** Uses `RecursiveCharacterTextSplitter` with semantic separators (e.g., "Section", "Clause") to preserve legal context.
* **Citation Engine:** The system prompt forces the AI to cite specific clauses or statutes, reducing hallucination risks.

Tech Stack

* **Frontend:** Streamlit
* **Orchestration:** LangChain
* **LLM:** Google Gemini 2.5 Pro (via `langchain-google-genai`)
* **Embeddings:** Google Generative AI Embeddings (`models/embedding-001`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Data Processing:** `pypdf`, `docx2txt`

Architecture

The application follows a specialized RAG workflow:

1.  Ingestion: User uploads a lease.
    System loads pre-indexed State Law text (NY/CA).
2.  Splitting: Documents are chunked into 500-character segments with 100-character overlap to maintain context window efficiency.
3.  Indexing: Two separate FAISS vector stores are created: `lease_db` and `law_db`.
4.  Retrieval:  When a user asks a question, the system performs a similarity search on BOTH databases ($k=4$ for each).
5.  Generation: The 8 retrieved chunks are concatenated into a single context block.
    Gemini Pro generates an answer based strictly on that context.

Installation & Setup

1.  Clone the repository
    ```bash
    git clone [https://github.com/yourusername/student-lease-advisor.git](https://github.com/yourusername/student-lease-advisor.git)
    cd student-lease-advisor
    ```

2.  Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```

3.  Set up API Keys
    Create a `.env` file or export your Google API key:
    ```bash
    export GOOGLE_API_KEY="your_api_key_here"
    ```

4.  Run the App
    ```bash
    streamlit run app.py
    ```

Project Structure

```text
├── app.py             
├── requirements.txt    
├── README.md              
└── data/


https://github.com/Lokeshkanna7/ClauseBreaker/blob/main/acrhitect_diagram.png
