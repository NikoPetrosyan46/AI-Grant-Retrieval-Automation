import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

def fetch_all_data():
    """
    Combines Zoho and Google Sheets. 
    Crucial: Every Document now includes a 'status' in metadata.
    """
    # This is a representative sample of how your data should look
    # once pulled from your Zoho/Google Sheets functions.
    processed_docs = [
        Document(
            page_content="STEM Education Grant 2024. Focused on providing laptops to rural schools. Budget: $25,000.",
            metadata={"source": "Zoho CRM", "id": "G-99", "status": "Approved"}
        ),
        Document(
            page_content="Urban Garden Initiative. Community-led project for sustainable food sources. Budget: $10,000.",
            metadata={"source": "Google Sheets", "date": "2026-03-01", "status": "Pending"}
        )
    ]
    return processed_docs

def initialize_system():
    """Sets up the RAG pipeline on server startup."""
    print("Indexing grant materials and initializing local LLM...")
    
    docs = fetch_all_data()
    
    # Using local embeddings (Nomic is highly efficient for retrieval)
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    
    # Persistence allows the database to stay saved on your drive
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory="./grant_db"
    )
    
    # We increase k to 5 to give the AI more context for summaries
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return create_chatbot_chain(retriever)

def create_chatbot_chain(retriever):
    """Defines the 'personality' and reasoning logic of the AI."""
    llm = Ollama(model="llama3")

    # The System Prompt is the key to the summarization feature.
    # It tells the AI exactly how to format different types of requests.
    system_prompt = (
        "You are an expert Grant Management System. "
        "Use the provided context to fulfill the user's request.\n\n"
        "GUIDELINES:\n"
        "1. If the user asks to 'summarize', provide a clear 3-bullet point summary "
        "covering: Goal, Funding, and Current Status.\n"
        "2. If the user asks a general question, provide a concise answer based ONLY on the context.\n"
        "3. Always mention the 'Status' of the grant if it is available in the context.\n"
        "4. If the answer is not in the context, say: 'Information not found in internal records.'\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # create_stuff_documents_chain handles the 'stuffing' of retrieved text into the prompt
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    # The final chain that connects the Retriever to the LLM
    return create_retrieval_chain(retriever, combine_docs_chain)
