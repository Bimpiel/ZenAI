from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import chromadb
import openai
from dotenv import load_dotenv
import os

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

    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    chat_history.append({"role": "user", "content": user_input})

    # Using the new OpenAI API interface
    response = openai.completions.create(
        model="gpt-4",  # Use the appropriate model (adjust as needed)
        messages=chat_history,
    )

    ai_response = response['choices'][0]['message']['content']
    chat_history.append({"role": "assistant", "content": ai_response})

    return jsonify({"response": ai_response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if no PORT env variable
    app.run(debug=True, host="0.0.0.0", port=port)
