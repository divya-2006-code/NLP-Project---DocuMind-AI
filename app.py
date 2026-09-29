
import os
import streamlit as st
import pymupdf

from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# Load API key
load_dotenv(
    dotenv_path=Path(__file__).resolve().parent / ".env",
    override=True
)

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)


# Load embedding model
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Extract text from multiple PDFs
def extract_pdf_text(uploaded_files):

    all_chunks = []
    all_metadatas = []
    document_details = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    for uploaded_file in uploaded_files:

        pdf_document = pymupdf.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf"
        )

        total_pages = len(pdf_document)

        for page_number, page in enumerate(pdf_document):

            page_text = page.get_text()

            if page_text.strip():

                page_chunks = text_splitter.split_text(page_text)

                for chunk in page_chunks:

                    all_chunks.append(chunk)

                    all_metadatas.append({
                        "source": uploaded_file.name,
                        "page": page_number + 1
                    })

        pdf_document.close()

        document_details.append({
            "name": uploaded_file.name,
            "pages": total_pages,
            "size": round(uploaded_file.size / 1024, 2)
        })

    return all_chunks, all_metadatas, document_details


# Create FAISS database
def create_vector_database(chunks, metadatas):

    embeddings = load_embedding_model()

    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vectorstore


# Generate answer using Groq
def generate_answer(question, vectorstore):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "Groq API key not found. Check your .env file."
        )

    retrieved_docs = vectorstore.similarity_search(
        question,
        k=3
    )

    context = ""

    for doc in retrieved_docs:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        context += (
            f"\nSource: {source}, Page: {page}\n"
            f"{doc.page_content}\n"
        )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=api_key
    )

    prompt = f"""
You are DocuMind AI, a PDF question-answering assistant.

Answer using only the information in the PDF context.

If the answer is not available, say:
"The answer was not found in the uploaded documents."

Do not invent facts.

PDF Context:
{context}

User Question:
{question}

Provide a clear and simple answer.
"""

    response = llm.invoke(prompt)

    return response.content, retrieved_docs


# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar
st.sidebar.title("📚 DocuMind AI")

st.sidebar.write(
    "Upload PDF documents and ask questions "
    "about their contents."
)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

st.sidebar.divider()

st.sidebar.subheader("Your Documents")

if uploaded_files:

    for file in uploaded_files:
        st.sidebar.write("📄", file.name)

    st.sidebar.success(
        f"{len(uploaded_files)} PDF(s) selected"
    )

else:
    st.sidebar.info("No PDFs uploaded yet.")

st.sidebar.divider()

if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()


# Main page
st.title("📚 DocuMind AI")

st.subheader(
    "PDF Question Answering System using NLP and RAG"
)

st.write(
    "Upload one or more PDFs and ask questions "
    "in simple English."
)

st.divider()


# Process uploaded PDFs
if uploaded_files:

    try:

        with st.spinner("Reading PDF documents..."):

            chunks, metadatas, document_details = (
                extract_pdf_text(uploaded_files)
            )

        if not chunks:

            st.warning(
                "No readable text found in these PDFs. "
                "They may contain scanned images."
            )

        else:

            with st.spinner(
                "Creating embeddings and FAISS database..."
            ):

                vectorstore = create_vector_database(
                    chunks,
                    metadatas
                )

            st.success(
                f"{len(uploaded_files)} PDF(s) are ready!"
            )

            # Document details
            with st.expander("📄 View Document Details"):

                for document in document_details:

                    st.write(
                        f"File: {document['name']}"
                    )

                    st.write(
                        f"Pages: {document['pages']}"
                    )

                    st.write(
                        f"Size: {document['size']} KB"
                    )

                    st.divider()

            # Chat section
            st.header("💬 Ask Your Documents")

            st.caption(
                "Your previous questions and answers "
                "will appear below."
            )

            # Display previous conversations
            for chat in st.session_state.chat_history:

                with st.chat_message("user"):
                    st.write(chat["question"])

                with st.chat_message("assistant"):

                    st.write(chat["answer"])

                    with st.expander("View Sources"):

                        for source in chat["sources"]:

                            st.write(
                                f"📄 {source['file']} "
                                f"— Page {source['page']}"
                            )

                            st.caption(source["text"])

            # Ask a new question
            question = st.chat_input(
                "Ask a question about your PDFs..."
            )

            if question:

                with st.chat_message("user"):
                    st.write(question)

                with st.chat_message("assistant"):

                    with st.spinner(
                        "Searching PDFs and generating answer..."
                    ):

                        try:

                            answer, retrieved_docs = (
                                generate_answer(
                                    question,
                                    vectorstore
                                )
                            )

                            st.write(answer)

                            sources = []

                            for doc in retrieved_docs:

                                sources.append({
                                    "file": doc.metadata.get(
                                        "source", "Unknown"
                                    ),
                                    "page": doc.metadata.get(
                                        "page", "Unknown"
                                    ),
                                    "text": doc.page_content
                                })

                            with st.expander("View Sources"):

                                for source in sources:

                                    st.write(
                                        f"📄 {source['file']} "
                                        f"— Page {source['page']}"
                                    )

                                    st.caption(source["text"])

                            # Save conversation
                            st.session_state.chat_history.append({
                                "question": question,
                                "answer": answer,
                                "sources": sources
                            })

                        except Exception as e:

                            st.error(
                                "Error while generating answer."
                            )

                            st.write(str(e))

            # Download chat history
            st.divider()

            st.subheader("⬇️ Download Your Answers")

            if st.session_state.chat_history:

                download_text = (
                    "DocuMind AI - Chat History\n"
                    + "=" * 40 + "\n\n"
                )

                for index, chat in enumerate(
                    st.session_state.chat_history,
                    start=1
                ):

                    download_text += (
                        f"Question {index}:\n"
                        f"{chat['question']}\n\n"
                    )

                    download_text += (
                        f"Answer:\n{chat['answer']}\n\n"
                    )

                    download_text += "Sources:\n"

                    for source in chat["sources"]:

                        download_text += (
                            f"- {source['file']}, "
                            f"Page {source['page']}\n"
                        )

                    download_text += (
                        "\n" + "-" * 40 + "\n\n"
                    )

                st.download_button(
                    label="Download Chat History",
                    data=download_text,
                    file_name="documind_chat_history.txt",
                    mime="text/plain"
                )

            else:

                st.info(
                    "Ask a question first to download "
                    "your answers."
                )

    except Exception as e:

        st.error("Error while processing PDFs.")
        st.write(str(e))

else:

    st.info(
        "Upload one or more PDF files from the "
        "sidebar to get started."
    )

    st.markdown("""
    ### Features

    - Upload multiple PDF documents.
    - Ask questions using natural language.
    - View previous questions and answers.
    - Check source PDF names and page numbers.
    - Download chat history as a text file.
    """)