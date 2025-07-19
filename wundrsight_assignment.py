
# !pip install -qU langchain_community pypdf langchain_huggingface
# !pip install -qU sentence-transformers faiss-cpu langchain langchain-groq python-dotenv

import torch
import faiss
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

def load_all_documents(folder_path: str):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs


# 3. Chunking
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_documents(docs)

all_docs = load_all_documents("./docs")

chunks = split_documents(all_docs)

print(len(all_docs))
print(len(chunks))

# 4. Embeddings
def build_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)


embedding = build_embeddings()

vectordb = FAISS.from_documents(chunks, embedding)
# Save the FAISS index to disk
save_path = "faiss_index"
vectordb.save_local(save_path)
print(f"FAISS index saved to {save_path}")

# print(vectordb)

query = input("Enter your query: ")
retrieved_docs = vectordb.similarity_search(query, k=5)

context = "\n\n\n".join([doc.page_content for doc in retrieved_docs])
print("Context:\n", context)

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


# Load into Langchain prompt
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create LLM
llm = ChatGroq(
    temperature=0.15,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    model_kwargs={"max_completion_tokens": 1024}
)

# Build RAG pipeline with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_PROMPT}
)

query = input("Ask your question: ")
response = qa_chain.invoke({"query":query})
response

print(response['result'])

