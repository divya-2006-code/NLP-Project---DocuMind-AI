
import os
import streamlit as st
import pymupdf

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


load_dotenv()

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocuMind AI")
st.subheader("PDF Question Answering System using RAG")

st.write(
    "Upload a PDF document and ask questions "
    "about its contents."
)

st.divider()

st.header("Upload Your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success("PDF uploaded successfully!")

    st.write("File name:", uploaded_file.name)

    st.write(
        "File size:",
        round(uploaded_file.size / 1024, 2),
        "KB"
    )

    pdf_document = pymupdf.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    extracted_text = ""
    page_texts = []

    for page_number, page in enumerate(pdf_document):

        page_text = page.get_text()

        page_texts.append(
            {
                "text": page_text,
                "page": page_number + 1
            }
        )

        extracted_text += (
            f"\n\nPage {page_number + 1}\n"
            f"{page_text}"
        )

    pdf_document.close()

    if extracted_text.strip():

        st.success("Text extracted successfully!")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = []
        metadatas = []

        for page_data in page_texts:

            page_chunks = text_splitter.split_text(
                page_data["text"]
            )

            for chunk in page_chunks:
                chunks.append(chunk)

                metadatas.append(
                    {
                        "page": page_data["page"],
                        "source": uploaded_file.name
                    }
                )

        st.success(
            "Text divided into chunks successfully!"
        )

        st.write(
            "Total number of chunks:",
            len(chunks)
        )

        with st.expander("View Text Chunks"):

            for i, chunk in enumerate(chunks):

                st.write(f"Chunk {i + 1}")
                st.write(chunk)
                st.divider()

        st.header("Create Embeddings and FAISS Database")

        @st.cache_resource
        def load_embedding_model():

            return HuggingFaceEmbeddings(
                model_name=(
                    "sentence-transformers/"
                    "all-MiniLM-L6-v2"
                )
            )

        with st.spinner(
            "Creating embeddings and FAISS database..."
        ):

            embeddings = load_embedding_model()

            vectorstore = FAISS.from_texts(
                texts=chunks,
                embedding=embeddings,
                metadatas=metadatas
            )

        st.success(
            "Embeddings created and FAISS database is ready!"
        )

        st.write(
            "Total chunks stored:",
            len(chunks)
        )

        st.divider()

        st.header("Ask Questions About Your PDF")

        question = st.text_input(
            "Enter your question"
        )

        if st.button("Ask DocuMind AI"):

            if question.strip():

                api_key = os.getenv("GROQ_API_KEY")

                if not api_key:

                    st.error(
                        "Groq API key not found. "
                        "Please check your .env file."
                    )

                else:

                    try:

                        with st.spinner(
                            "Searching your PDF and generating an answer..."
                        ):

                            retrieved_docs = (
                                vectorstore.similarity_search(
                                    question,
                                    k=3
                                )
                            )

                            context = "\n\n".join(
                                [
                                    f"Page {doc.metadata.get('page', 'Unknown')}:\n"
                                    f"{doc.page_content}"
                                    for doc in retrieved_docs
                                ]
                            )

                            llm = ChatGroq(
                                    model="openai/gpt-oss-120b",
                                    temperature=0,
                                    api_key=api_key
                            )

                            prompt = f"""
You are DocuMind AI, a PDF question-answering assistant.

Answer the user's question using only the context
retrieved from the uploaded PDF.

If the answer is not available in the context,
say:
"The answer was not found in the uploaded document."

Do not invent facts.

Context from the PDF:
{context}

User question:
{question}

Provide a clear and simple answer.
"""

                            response = llm.invoke(prompt)

                            st.subheader("Answer")

                            st.write(response.content)

                            st.subheader("Sources")

                            for doc in retrieved_docs:

                                page_number = doc.metadata.get(
                                    "page",
                                    "Unknown"
                                )

                                st.write(
                                    f"Page {page_number}"
                                )

                                st.caption(
                                    doc.page_content
                                )

                    except Exception as e:

                        st.error(
                            "An error occurred while "
                            "generating the answer."
                        )

                        st.write(str(e))

            else:

                st.warning(
                    "Please enter a question first."
                )

        st.divider()

        st.header("Extracted PDF Text")

        st.text_area(
            "Document contents",
            value=extracted_text,
            height=350
        )

    else:

        st.warning(
            "No readable text found in this PDF. "
            "It may contain scanned images."
        )