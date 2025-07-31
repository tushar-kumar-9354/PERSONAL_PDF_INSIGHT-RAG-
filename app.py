import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
assert api_key, "GOOGLE_API_KEY not found in .env"

# Streamlit UI
st.title("RAG APPLICATION USING GEMINI")

# Load PDF
loader = PyPDFLoader("medi.pdf")
data = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)
st.write(f"Loaded {len(docs)} chunks from PDF.")

# Embedding + Vector Store
import asyncio

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Initialize Embeddings Model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = Chroma.from_documents(docs, embedding_model, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # ✅ use valid model name
    temperature=0.3,
    max_tokens=500
)

# Prompt Template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Chat Input
query = st.chat_input("Ask Me Anything ...")
if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})

    st.write("**Question:**")
    st.write(query)
    st.write("**Answer:**")
    st.write(response["answer"])
