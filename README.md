# pdf-rag-chat

This is a small Streamlit app that lets you upload a PDF and ask questions about its content. Under the hood it follows a fairly standard RAG (retrieval-augmented generation) setup: the text from the PDF is split into chunks, indexed using embeddings in a FAISS vector store, and when you ask a question the most relevant chunks are retrieved and passed as context to a language model to generate the answer.

One deliberate design choice is that embeddings are generated locally using HuggingFace sentence transformers. This means indexing a document does not require any external API calls or cost. OpenAI is only used at the final step, when generating the natural-language answer based on the retrieved context.

---

## Running the project

Clone the repository and move into the project directory:

```bash
git clone https://github.com/SimonC03/pdf-rag-chat.git
cd pdf-rag-chat