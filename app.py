import os
import hashlib
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import io

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


@st.cache_resource(show_spinner=False)
def get_local_embeddings():
    # Local embeddings => no OpenAI cost for indexing
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def build_faiss_index(file_id: str, chunks: list[str]):
    embeddings = get_local_embeddings()
    return FAISS.from_texts(chunks, embeddings)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))  # <-- FIX
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts).strip()


def main():
    st.set_page_config(page_title="Chatta med din PDF", page_icon="📚")
    st.header("Chatta med din PDF (RAG DEMO)")

    with st.sidebar:
        st.subheader("Inställningar")
        k = st.slider("Antal textbitar (k) att hämta", 1, 10, 4)
        show_sources = st.checkbox("Visa källor (chunks)", value=False)

    pdf = st.file_uploader("Ladda upp din PDF-fil", type="pdf")

    if pdf is None:
        st.info("Ladda upp en PDF för att börja.")
        return

    pdf_bytes = pdf.getvalue()
    pdf_id = file_hash(pdf_bytes)

    # Extract & chunk text (cache in session to avoid rework on reruns)
    if st.session_state.get("pdf_id") != pdf_id:
        st.session_state.pdf_id = pdf_id
        st.session_state.text = ""
        st.session_state.chunks = []
        st.session_state.index = None

        with st.spinner("Läser PDF och delar upp text..."):
            text = extract_text_from_pdf(pdf_bytes)

            if not text:
                st.warning("Kunde inte extrahera text från PDF:en (kan vara skannad bild-PDF).")
                return

            splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = splitter.split_text(text)

            st.session_state.text = text
            st.session_state.chunks = chunks

        with st.spinner("Bygger lokal sökindex (FAISS)..."):
            try:
                st.session_state.index = build_faiss_index(pdf_id, st.session_state.chunks)
            except Exception as e:
                st.error(f"Kunde inte bygga sökindex: {e}")
                return

    knowledge_base = st.session_state.index
    retriever = knowledge_base.as_retriever(search_kwargs={"k": k})

    user_question = st.text_input("Ställ din fråga om PDF-innehållet:")

    if not user_question:
        return

    # ChatOpenAI uses OPENAI_API_KEY; indexing is local but answering still needs a key.
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY saknas. Lägg den i en .env-fil eller exportera den i terminaln.")
        return

    llm = ChatOpenAI(temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Du är en hjälpsam assistent. Svara på svenska. "
             "Använd endast kontexten. Om svaret inte finns i kontexten, säg det tydligt."),
            ("human", "KONTEXT:\n{context}\n\nFRÅGA:\n{question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    with st.spinner("Hämtar svar..."):
        try:
            docs = retriever.invoke(user_question)
            context = "\n\n".join(d.page_content for d in docs)

            answer = chain.invoke({"context": context, "question": user_question})
            st.write("**Svar:**")
            st.write(answer)

            if show_sources:
                st.divider()
                st.write("**Källor (chunks):**")
                for i, d in enumerate(docs, 1):
                    with st.expander(f"Chunk {i}"):
                        st.write(d.page_content)

        except Exception as e:
            # Most common: quota / invalid key / network
            st.error(f"Något gick fel vid API-anropet: {e}")
            st.info(
                "Om felet är 429 (insufficient_quota): lägg till billing på OpenAI Platform "
                "eller byt även LLM till en lokal modell."
            )


if __name__ == "__main__":
    main()