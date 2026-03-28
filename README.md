# AI-Grant-Retrieval-Automation
AI grant retrieval code

To run this, you would need to install these libraries: pip install langchain langchain-community chromadb requests pandas


The Workflow:

Backend: Flask handles the web requests.

Logic: LangChain connects to Ollama (local LLM) and ChromaDB (vector store).

Frontend: A simple index.html file using JavaScript fetch() to talk to your Flask route.

Key Python Libraries to use:

flask: To create the web server.

langchain_community: To use the Ollama and Chroma classes.

pandas: To handle the Google Sheets CSV data easily.
