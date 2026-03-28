# rag_logic.py
import os
import pandas as pd
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def initialize_system():
    """Sets up the RAG pipeline on startup."""
    # 1. Ingest (Using your real SHEET_URL from before)
    # zoho_docs = fetch_zoho_grant_data()
    # sheet_docs = fetch_google_sheets_intake(SHEET_URL)
    # all_docs = zoho_docs + sheet_docs
    
    # Placeholder for local testing if you don't have docs yet
    from langchain_core.documents import Document
    all_docs = [Document(page_content="The STEM grant provides $5k for tech.", metadata={"source":"test"})]

    # 2. Build Vector Store
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    vectorstore = Chroma.from_documents(documents=all_docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # 3. Create Chain
    llm = Ollama(model="llama3")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grant assistant. Use context: {context}"),
        ("human", "{input}"),
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)
