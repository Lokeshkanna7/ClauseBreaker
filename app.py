import streamlit as st
import tempfile
import os
import time
from dotenv import load_dotenv


load_dotenv(override=True)


from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


st.set_page_config(
    page_title="Student Lease Break Advisor",
    layout="wide"
)

st.title("Student Lease Break Advisor")
st.caption("Legal information assistant for students (not legal advice)")


def get_api_key():
    # 1. Try Streamlit Secrets (Cloud Deployment)
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except FileNotFoundError:
        pass 


    env_key = os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key
    
    return None


api_key = get_api_key()

if not api_key:
    st.error("🚨 API Key missing! On Streamlit Cloud, add it to 'Secrets'. Locally, use .env")
    st.stop()



def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "Section", "Clause"]
    )

def create_vectorstore_with_batching(chunks, embeddings):
    """
    Embeds documents in small batches with pauses to respect Google's
    free tier rate limits (avoiding 429 errors).
    """
    batch_size = 5  
    vectorstore = None
    
    progress_text = "Embedding document chunks... Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, batch_size):
        
        batch = chunks[i : i + batch_size]
        
        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
        except Exception as e:
            st.error(f"Error on batch {i}: {e}")
            return None

        
        percent_complete = min((i + batch_size) / total_chunks, 1.0)
        my_bar.progress(percent_complete, text=f"Processing chunk {min(i + batch_size, total_chunks)} of {total_chunks}")
        
        
        time.sleep(2)
        
    
    my_bar.empty()
    return vectorstore

@st.cache_resource
def process_uploaded_file(file_content, file_name):
    """
    Saves uploaded file to temp disk so LangChain can read it,
    then embeds it into a vector database using batching.
    """
    ext = os.path.splitext(file_name)[1].lower()
    
  
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
       
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            return None

      
        docs = loader.load()
        splitter = get_splitter()
        chunks = splitter.split_documents(docs)

      
        for d in chunks:
            d.metadata["source"] = "lease"
            
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", # Newer model, often more stable
            google_api_key=api_key
        )
        
       
        return create_vectorstore_with_batching(chunks, embeddings)

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

    finally:
       
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

@st.cache_resource
def get_law_vectorstore(state):
    """
    Loads state-specific laws into a separate vector database.
    """
    laws = {
        "New York": """
        New York Real Property Law §227-e (Duty to Mitigate): Landlords must make reasonable efforts to re-rent the apartment if a tenant vacates early. If they re-rent it, the original tenant's lease terminates.
        New York Real Property Law §227-a: Senior citizens or those moving for medical reasons have specific rights to terminate early.
        New York Real Property Law §235-b (Warranty of Habitability): Landlords must maintain the property in a safe, livable condition.
        """,
        "California": """
        California Civil Code §1951.2: Landlords can recover unpaid rent but must prove they made reasonable efforts to minimize damages by re-renting the unit.
        California Civil Code §1942: If a unit is uninhabitable (no heat, water, etc.), a tenant may repair and deduct or vacate the premises (constructive eviction).
        """
    }

 
    content = laws.get(state, "State law information unavailable.")
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt", encoding='utf-8') as f:
        f.write(content)
        law_path = f.name

    try:
        loader = TextLoader(law_path, encoding='utf-8')
        docs = loader.load()
        splitter = get_splitter()
        chunks = splitter.split_documents(docs)

        for d in chunks:
            d.metadata["source"] = "law"
            d.metadata["state"] = state

       
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
        
       
        return create_vectorstore_with_batching(chunks, embeddings)
    
    finally:
        if os.path.exists(law_path):
            try:
                os.remove(law_path)
            except:
                pass

def generate_answer(question, lease_db, law_db):
    
    lease_docs = lease_db.similarity_search(question, k=4)
    law_docs = law_db.similarity_search(question, k=2)

    
    context_text = "\n\n".join(
        [f"[{d.metadata['source'].upper()}] {d.page_content}" 
         for d in lease_docs + law_docs]
    )

   
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0,
        google_api_key=api_key
    )

    prompt = f"""
    You are a legal information assistant for students.
    You do NOT provide legal advice.
    You must base your response strictly on the provided lease clauses and state laws.
    
    Question: {question}
    
    Context:
    {context_text}
    
    Respond with:
    1. Whether early termination may be legally allowed based on the text.
    2. Required notice period (if mentioned).
    3. Potential penalties or risks.
    4. Explicit citations (Section numbers or Statute names).
    5. A clear legal disclaimer at the end.
    """
    
    return llm.invoke(prompt).content


state = st.sidebar.selectbox("Select U.S. State", ["New York", "California"])

uploaded_file = st.sidebar.file_uploader(
    "Upload Lease (PDF or DOCX)", 
    type=["pdf", "docx"]
)

st.sidebar.markdown("---")
st.sidebar.info("**Disclaimer:** This tool extracts text from your lease and compares it to general state laws. It is not a substitute for a lawyer.")


if uploaded_file:
    
    lease_db = process_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
    law_db = get_law_vectorstore(state)
    
    if lease_db:
        st.success("Analysis Complete. You may ask questions below.")
        
       
        question = st.chat_input("Example: Can I break my lease if I found a new job?")
        
        if question:
           
            with st.chat_message("user"):
                st.write(question)
            
            
            with st.chat_message("assistant"):
                with st.spinner("Consulting lease terms and state laws..."):
                    try:
                        response = generate_answer(question, lease_db, law_db)
                        st.write(response)
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
    else:
        st.error("Failed to process document. Please try a different PDF.")

else:

    st.info("👋 Welcome! Please upload your lease agreement on the left sidebar to begin.")
