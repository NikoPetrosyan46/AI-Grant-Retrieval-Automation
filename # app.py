# app.py
from flask import Flask, render_template, request, jsonify
from rag_logic import initialize_system

app = Flask(__name__)

# Initialize the AI once when the server starts
print("Starting AI System... please wait.")
rag_chain = initialize_system()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.json.get('message')
    
    # Run the AI logic
    response = rag_chain.invoke({"input": user_message})
    
    # Return the answer and the sources as JSON
    return jsonify({
        "answer": response["answer"],
        "sources": [doc.metadata.get('source') for doc in response["context"]]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
