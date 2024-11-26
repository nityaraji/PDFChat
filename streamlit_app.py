import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber

from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFDirectoryLoader 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_together import Together
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_together import Together
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

st.set_page_config(
    page_title="Together PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=True)
def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    # logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    try:
        path = os.path.join(temp_dir, file_upload.name)
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())
            # logger.info(f"File saved to temporary path: {path}")
            loader = PyMuPDFLoader(path)
            data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        # logger.info("Document split into chunks")

        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name='BAAI/bge-base-en-v1.5',
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        persist_dir = "./storage"
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="myRAG",
            persist_directory=persist_dir
        )
        # logger.info("Vector DB created")

    except Exception as e:
        # logger.error(f"Error during vector DB creation: {e}")
        raise e
    finally:
        shutil.rmtree(temp_dir)
        # logger.info(f"Temporary directory {temp_dir} removed")
    
    return vector_db



def process_question(question: str, vector_db: Chroma) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    # logger.info(f"Processing question: {question}")
    
    llm = Together(model='mistralai/Mistral-7B-Instruct-v0.2',
                       together_api_key=os.getenv('TOGETHER_API_KEY'),
                       temperature=0.1,
                       top_p=0.8,
                       max_tokens=4096                       
                    )

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 10
        different versions of the given user question to retrieve relevant documents from
        a vector database.Make sure that input token limit do not exceed 28000 tokens. By 
        generating multiple perspectives on the user question, your goal is to help the user 
        overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    # logger.info("Question processed and response generated")
    return response


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    # logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    # logger.info("PDF pages extracted as images")
    return pdf_pages


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    # logger.info("Deleting vector DB")
    try:
        if vector_db is not None:
            vector_db.delete_collection()
            st.session_state["vector_db"] = None
            st.success("Collection and temporary files deleted successfully.")
            # logger.info("Vector DB and related session state cleared")
            st.rerun()
        else:
            st.error("No vector database found to delete.")
            # logger.warning("Attempted to delete uninitialized collection.")
    except ValueError as e:
            st.error(f"Error: {e}")
            # logger.error(f"ValueError during delete_collection: {e}")
    else:
        st.warning("No vector database found to delete.")
        # logger.warning("Attempted to delete vector DB, but none was found.")

class Request(BaseModel):
    prompt : str

class Response(BaseModel):
    response : str

def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.subheader("🧠 PDF RAG playground", divider="gray", anchor=False)

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
  
    file_upload = col1.file_uploader(
        "Upload a PDF file ↓",
        type="pdf",
        accept_multiple_files=False,
        key="pdf_uploader"
    )

    if file_upload:
        if st.session_state["vector_db"] is None:
            with st.spinner("Processing uploaded PDF..."):
                st.session_state["vector_db"] = create_vector_db(file_upload)
                pdf_pages = extract_all_pages_as_images(file_upload)
                st.session_state["pdf_pages"] = pdf_pages

    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        zoom_level = col1.slider(
            "Zoom Level", 
            min_value=100, 
            max_value=1000, 
            value=700, 
            step=50,
            key="zoom_slider"
        )

        with col1:
            with st.container(height=410, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)

    delete_collection = col1.button(
        "⚠️ Delete collection", 
        type="secondary",
        key="delete_button"
    )

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    with col2:
        message_container = st.container(height=500, border=True)

        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"]
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")

                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file to begin chat...")


if __name__ == "__main__":
    main()