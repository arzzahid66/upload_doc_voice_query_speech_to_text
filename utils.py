import streamlit as st
import os
from langchain_community.vectorstores import Qdrant
import openai
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
def upload_file_and_preprosses():
    os.makedirs("uploads", exist_ok=True)
    file_path = None  # Initialize file_path
    try:
        uploaded_file = st.file_uploader("Upload a file", type=["pdf", "csv", "docx", "txt"])
        if st.button("submit"):
            if uploaded_file is not None:
                file_path = os.path.join("uploads", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_extension = uploaded_file.name.split(".")[-1]

                if file_extension == "pdf":
                    loader = PyMuPDFLoader(file_path)
                    pages = loader.load()
                elif file_extension == "csv":
                    loader = CSVLoader(file_path=file_path)
                    pages = loader.load()
                elif file_extension == "docx":
                    file_loader = Docx2txtLoader(file_path)
                    pages = file_loader.load()
                else:
                    st.error("Unsupported file format.")

                def split_doc(pages):
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=700,
                        chunk_overlap=50,
                        length_function=len,
                        is_separator_regex=False)
                    texts = text_splitter.split_documents(pages)
                    return texts
               
                text_new = split_doc(pages)
                qdrant_key = os.getenv("QDRANT_KEY")
                URL = os.getenv("URL")
                qdrant = Qdrant.from_documents(
                    text_new,
                    embedding_model,
                    url=URL,
                    prefer_grpc=False,
                    api_key=qdrant_key,
                    collection_name = "voicequery",
                    force_recreate=True)
                
                st.info("file uploaded successfully...")

    except Exception as ex:
        st.error(f"Error :{ex}")
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
def qdrant_client():
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        qdrant_key = os.getenv("QDRANT_KEY")
        URL = os.getenv("URL")
        qdrant_client = QdrantClient(
        url=URL,
        api_key=qdrant_key,
        )
        qdrant_store = Qdrant(qdrant_client,"voicequery" ,embedding_model)
        return qdrant_store

qdrant_store = qdrant_client()

from langchain.load import load,dumps
def get_unique_documents(docs:list):
    unique_docs = list(set([dumps(doc) for doc in docs]))
    return [load(doc) for doc in unique_docs]
     
def qa_ret(qdrant_store,input_query):
    try:
        template = """
        You are a helpful and dedicated female assistantt. Your primary role is to assist the user by providing accurate 
        and thoughtful answers based on the given context. If the user asks any questions related to the provided 
        information, respond in a courteous and professional manner.
        It is important to always give your best effort, as your assistance plays a crucial role in the user’s 
        career success. Ensure that your responses are clear, concise, and well-structured.
        **Note:** Always provide your responses in Roman Urdu (Hinglish) to maintain the preferred language format.
        {context}
        **Question:** {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        retriever= qdrant_store.as_retriever(search_type="similarity", search_kwargs={"k":4})
        setup_and_retrieval = RunnableParallel(
                {"context": retriever | get_unique_documents , "question": RunnablePassthrough()})
        
        model = ChatOpenAI(model = "gpt-4o-mini", openai_api_key = OPENAI_API_KEY, temperature=0.3)
        output_parser= StrOutputParser()
        rag_chain = (
        setup_and_retrieval
        | prompt
        | model
        | output_parser)
        respone=rag_chain.invoke(input_query)
        return respone
    except Exception as ex:
        return ex
    
def qa_retrieval(input_query):
    result = qa_ret(qdrant_store,input_query)
    return result
