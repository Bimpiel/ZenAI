from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import chromadb
import openai
from dotenv import load_dotenv
import os
from textblob import TextBlob
import nltk

# Download NLTK data for TextBlob (first-time only)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
CORS(app)

# Database setup
CHROMA_PATH = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="mental_docs")

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Retrieve relevant data
retrieved_data = collection.query(
    query_texts=["general mental health support"],
    n_results=3  # Reduced to save memory
)
retrieved_data = retrieved_data.get("documents", ["No relevant data found."])

# System prompt template
SYSTEM_PROMPT = f"""
You are a mental health chatbot offering support. Guidelines:
1. Keep responses brief (1-2 sentences)
2. Adapt tone to user's sentiment (positive/neutral/negative)
3. Use simple emojis occasionally 😊
4. Never say "I'm sorry to hear that"
5. Only answer mental health questions

Retrieved knowledge: {retrieved_data}
"""

chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
MAX_HISTORY_LENGTH = 10

def analyze_sentiment(text):
    """Lightweight sentiment analysis (positive/neutral/negative)"""
    analysis = TextBlob(text[:500])  # Limit input size
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    return "neutral"

def adapt_response(response, sentiment):
    """Minimal response adaptation"""
    if sentiment == "positive":
        return response + " 😊"
    elif sentiment == "negative":
        return response + " 🫂"
    return response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400
    
    # Analyze sentiment (lightweight)
    sentiment = analyze_sentiment(user_input)
    
    # Update chat history
    chat_history.append({
        "role": "user",
        "content": f"[{sentiment}] {user_input}"  # Tag message with sentiment
    })
    
    # Trim history
    if len(chat_history) > MAX_HISTORY_LENGTH:
        chat_history = [chat_history[0]] + chat_history[-(MAX_HISTORY_LENGTH-1):]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=chat_history,
            max_tokens=100,
            temperature=0.7
        )
        
        ai_response = response['choices'][0]['message']['content']
        ai_response = adapt_response(ai_response, sentiment)
        
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