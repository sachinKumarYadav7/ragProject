# 📄 LLM-Powered Document QA System

A lightweight RAG-based app that answers questions from uploaded PDF documents using Meta’s LLaMA 4 model via the Groq API.

---

## 🚀 Features

- 📤 Upload multiple PDF files via Streamlit UI
- 🧩 Intelligent document chunking (400 chars, 50 overlap)
- 🧠 Embeddings via `sentence-transformers/all-MiniLM-L6-v2`
- ⚡ FAISS-based semantic retrieval
- 🤖 LLM answers with document page references
- 🖥️ GPU/CPU auto-detection
- ✅ .env-based secure Groq API access

---

### 📦 Why Use `RecursiveCharacterTextSplitter`

We use `RecursiveCharacterTextSplitter` with `chunk_size=400` and `chunk_overlap=50` to split documents while preserving meaning. It recursively breaks text by paragraphs, lines, and words, ensuring semantic coherence. This improves retrieval accuracy, prevents context loss at boundaries, and provides high-quality chunks for embedding and LLM-based answering.


## 🛠️ Tech Stack


| Component        | Tool/Lib                             |
|------------------|--------------------------------------|
| Frontend         | Streamlit                            |
| LLM              | Meta LLaMA 4 Scout (via Groq API)    |
| Embeddings       | HuggingFace Sentence Transformers    |
| Vector Database  | FAISS                                |
| Doc Handling     | LangChain + PyPDFLoader              |
| Env Management   | `python-dotenv`                      |

---

## 📋 Requirements

- Python 3.12+
- Groq API Key ([Sign up](https://groq.com/))
- Recommended: virtualenv or [uv](https://github.com/astral-sh/uv)

---

## 🔧 Installation

```bash
git clone <repo-url>
cd wundrsight_assignment

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your_groq_key" > .env


"for simple use in you refer the `Wundrsight_Assignment.ipynb` in ##colab"
