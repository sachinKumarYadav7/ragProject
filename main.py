import os
import torch
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# === Load environment variables ===
load_dotenv()

DOCS_PATH = "./docs"
os.makedirs(DOCS_PATH, exist_ok=True)

# === File uploader ===
def handle_file_upload():
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(DOCS_PATH, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
        st.success(f"Uploaded {len(uploaded_files)} file(s) successfully!")

# === PDF Loader ===
def load_all_documents(folder_path: str):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_documents(docs)

@st.cache_resource
def build_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

@st.cache_resource
def get_vectorstore():
    all_docs = load_all_documents(DOCS_PATH)
    chunks = split_documents(all_docs)
    embedding = build_embeddings()
    vectordb = FAISS.from_documents(chunks, embedding)
    return vectordb

# === Prompt Template ===
prompt_template = """
You are a helpful assistant. Answer the question based only on the provided context from documents.
If the answer is not in the context, respond with "The answer is not available in the provided documents."

Include the page number from the context as a reference in your answer, like this: (Page X).

Context:
{context}

Question:
{question}

Answer:
"""
QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

@st.cache_resource
# def build_qa_chain(vectordb):
#     llm = ChatGroq(
#         temperature=0.25,
#         model_name="meta-llama/llama-4-scout-17b-16e-instruct",
#         model_kwargs={"max_completion_tokens": 1024}
#     )
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=vectordb.as_retriever(),
#         chain_type="stuff",
#         chain_type_kwargs={"prompt": QA_PROMPT}
#     )

@st.cache_resource
def build_qa_chain(_vectordb):
    llm = ChatGroq(
        temperature=0.25,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        model_kwargs={"max_completion_tokens": 1024}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=_vectordb.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

# === Streamlit App UI ===
st.set_page_config(page_title="Mini RAG QA", layout="centered")
st.title("📄 LLM-Powered Document QA System")

# Check if docs exist
pdf_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".pdf")]
if not pdf_files:
    st.warning("No PDF files found in `/docs`. Please upload some documents.")
    handle_file_upload()
    st.stop()
else:
    st.success(f"Found {len(pdf_files)} document(s) in `/docs` folder.")

# Load Vector Store
vectordb = get_vectorstore()
qa_chain = build_qa_chain(vectordb)

# Question Input
query = st.text_input("🔎 Ask a question based on the documents:")

if query:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke({"query": query})
        st.markdown("### ✅ Answer:")
        st.write(response['result'])

        # Optional: Show top matching docs
        retrieved_docs = vectordb.similarity_search(query, k=3)
        with st.expander("📚 View Retrieved Context"):
            for i, doc in enumerate(retrieved_docs, 1):
                st.markdown(f"**Context {i}:**")
                st.markdown(doc.page_content[:1000])
