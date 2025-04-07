from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import chromadb
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Flask setup
app = Flask(__name__)
CORS(app)

# Database setup
CHROMA_PATH = r"chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="mental_docs")

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Retrieve relevant data once per session
retrieved_data = collection.query(query_texts=["general mental health support"], n_results=5)
retrieved_data = retrieved_data.get("documents", ["No relevant data found."])

# Global chat history
chat_history = [
    {"role": "system", "content": f"""
        You are a mental health chatbot offering support. Show empathy and use the retrieved data for accuracy. 
        Only answer with factual statements from the retrieved data.
        All answers should be related to mental health. If someone asks about anything unrelated, respond with "I don't know much."
        Keep your tone calm, friendly, and informal. Use emojis to make the chat more human! 😊
        Retrieved data: {retrieved_data}
    """}
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history

    # Get user input from the request
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Append the user message to chat history
    chat_history.append({"role": "user", "content": user_input})

    # Use OpenAI's ChatCompletion API to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-4",  # You can change this to another model like gpt-3.5-turbo
        messages=chat_history,  # Provide the chat history to maintain conversation context
    )

    # Extract the assistant's response from the API response
    ai_response = response['choices'][0]['message']['content']

    # Append the assistant's response to the chat history
    chat_history.append({"role": "assistant", "content": ai_response})

    # Return the response in JSON format
    return jsonify({"response": ai_response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to port 5000 if no PORT environment variable is set
    app.run(debug=True, host="0.0.0.0", port=port)
