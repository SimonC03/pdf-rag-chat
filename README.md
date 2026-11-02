# pdf-rag-chat

A small Streamlit app for asking questions about an uploaded PDF using retrieval-augmented generation (RAG). Embeddings run locally, so indexing a document needs no API calls. OpenAI is called only to write the final answer.

The interface and the answers are in Swedish.

## How it works

1. **Extract**: PyPDF2 reads the text of every page. Scanned (image-only) PDFs yield no text and are rejected with a warning.
2. **Chunk**: LangChain's `CharacterTextSplitter` splits the text on newlines into chunks of 1000 characters with 200 characters of overlap.
3. **Embed**: chunks are embedded locally with the HuggingFace sentence-transformers model `all-MiniLM-L6-v2`.
4. **Index**: the vectors go into an in-memory FAISS index, cached per file by its SHA-256 hash so reruns don't rebuild it.
5. **Retrieve**: the top *k* chunks for the question are fetched (k is 1–10, default 4, set in the sidebar).
6. **Answer**: the chunks and the question go to OpenAI through LangChain's `ChatOpenAI` (temperature 0). The prompt tells the model to answer only from that context and to say so when the answer isn't there. The retrieved chunks can be shown as sources.

## Tech stack

Python, Streamlit, LangChain, FAISS, sentence-transformers (HuggingFace), OpenAI API, PyPDF2.

## Getting started

Prerequisites: Python 3 and an OpenAI API key.

```bash
git clone https://github.com/SimonC03/pdf-rag-chat.git
cd pdf-rag-chat
pip install streamlit python-dotenv PyPDF2 langchain-community langchain-openai langchain-text-splitters faiss-cpu sentence-transformers
```

Set `OPENAI_API_KEY`, either in a `.env` file in the project root or as an exported environment variable. Then start the app:

```bash
streamlit run app.py
```

The first run downloads the embedding model from HuggingFace.
