from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import chromadb
import openai
from dotenv import load_dotenv
import os
from textblob import TextBlob
import nltk

# Download required NLTK data (first-time only)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
CORS(app)

# Database setup - using your original path
CHROMA_PATH = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="mental_docs")

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Retrieve relevant data (reduced from 5 to 3 to save memory)
retrieved_data = collection.query(
    query_texts=["general mental health support"],
    n_results=3
)
retrieved_data = retrieved_data.get("documents", ["No relevant data found."])

# System prompt (modified slightly for better memory usage)
SYSTEM_PROMPT = f"""
You are a mental health chatbot offering support. Show empathy and use the retrieved data for accuracy.
Keep responses short (1-2 sentences max). Use a calm, friendly tone with occasional emojis 😊
Avoid generic sympathy statements. Adapt responses based on user sentiment.

Retrieved data: {retrieved_data}
"""

# Global chat history with your original structure
chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
MAX_CHAT_HISTORY_LENGTH = 10  # Reduced from 15 to save memory

def analyze_sentiment(text):
    """Lightweight sentiment analysis using TextBlob"""
    analysis = TextBlob(text[:512])  # Limit input size for memory
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.3:
        return "positive"
    elif polarity < -0.3:
        return "negative"
    return "neutral"

def adapt_response(response, sentiment):
    """Your original tone adaptation logic"""
    if sentiment == "positive":
        return response + " 😊"
    elif sentiment == "negative":
        return response + " 🫂 You're not alone in this."
    return response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history

    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Analyze sentiment (lightweight)
    sentiment = analyze_sentiment(user_input)
    
    # Append user message with sentiment context
    chat_history.append({
        "role": "user", 
        "content": f"[{sentiment}] {user_input}"
    })

    # Trim history if needed (keep system message)
    if len(chat_history) > MAX_CHAT_HISTORY_LENGTH:
        chat_history = [chat_history[0]] + chat_history[-(MAX_CHAT_HISTORY_LENGTH-1):]

    try:
        # Using gpt-3.5-turbo as in your original code
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=chat_history,
            max_tokens=100,
            temperature=0.7
        )

        ai_response = response['choices'][0]['message']['content']
        
        # Apply sentiment adaptation
        ai_response = adapt_response(ai_response, sentiment)
        
        # Append assistant response
        chat_history.append({"role": "assistant", "content": ai_response})

        return jsonify({
            "response": ai_response,
            "sentiment": sentiment  # Optional for frontend
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Temporary service issue"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)