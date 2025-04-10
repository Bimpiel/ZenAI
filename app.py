from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import chromadb
import openai
from dotenv import load_dotenv
import os
from textblob import TextBlob
import nltk
from collections import deque

# Download required NLTK data
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
    n_results=3
)
retrieved_data = retrieved_data.get("documents", ["No relevant data found."])

# System prompt
SYSTEM_PROMPT = f"""
You are a mental health chatbot offering support. Show empathy and use the retrieved data for accuracy.
Keep responses short (1-2 sentences max). Use a calm, friendly tone with occasional emojis 😊
Avoid generic sympathy statements. Adapt responses based on user sentiment.

Retrieved data: {retrieved_data}
"""

# Conversation tracking
chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
sentiment_history = deque(maxlen=5)  # Track last 5 sentiments
MAX_CHAT_HISTORY_LENGTH = 10
SENTIMENT_WEIGHTS = [0.2, 0.25, 0.25, 0.2, 0.1]  # Weighted toward recent messages

def analyze_sentiment(text):
    """Analyze sentiment of individual messages"""
    analysis = TextBlob(text[:512])  # Limit input size
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.3:
        return "positive"
    elif polarity < -0.3:
        return "negative"
    return "neutral"

def analyze_conversation_sentiment():
    """Calculate weighted average sentiment of entire conversation"""
    if not sentiment_history:
        return "neutral"
    
    # Convert sentiments to numerical scores
    sentiment_scores = []
    for sentiment in sentiment_history:
        if sentiment == "positive":
            sentiment_scores.append(1.0)
        elif sentiment == "negative":
            sentiment_scores.append(-1.0)
        else:
            sentiment_scores.append(0.0)
    
    # Calculate weighted average
    weights = SENTIMENT_WEIGHTS[-len(sentiment_scores):]  # Use appropriate weights subset
    weighted_sum = sum(score * weight for score, weight in zip(sentiment_scores, weights))
    
    if weighted_sum > 0.3:
        return "positive"
    elif weighted_sum < -0.3:
        return "negative"
    return "neutral"

def adapt_response(response, immediate_sentiment, conversation_sentiment):
    """Dual-layer sentiment adaptation"""
    # Immediate reaction
    if immediate_sentiment == "positive":
        response += " 😊"
    elif immediate_sentiment == "negative":
        response = "I hear you... " + response + " 🫂"
    
    # Conversation context
    if conversation_sentiment == "negative" and immediate_sentiment != "positive":
        if any(word not in response.lower() for word in ["challeng", "difficult", "hard"]):
            response = "This seems really challenging... " + response
    elif conversation_sentiment == "positive" and immediate_sentiment != "negative":
        response = response.replace("😊", "🌟")  # More celebratory
    
    return response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history, sentiment_history

    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Analyze sentiments
    immediate_sentiment = analyze_sentiment(user_input)
    sentiment_history.append(immediate_sentiment)
    conversation_sentiment = analyze_conversation_sentiment()
    
    # Append user message with sentiment context
    chat_history.append({
        "role": "user", 
        "content": f"[{immediate_sentiment}] {user_input}"
    })

    # Trim history if needed (keep system message)
    if len(chat_history) > MAX_CHAT_HISTORY_LENGTH:
        chat_history = [chat_history[0]] + chat_history[-(MAX_CHAT_HISTORY_LENGTH-1):]

    try:
        # Generate response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=chat_history,
            max_tokens=100,
            temperature=0.7
        )

        ai_response = response['choices'][0]['message']['content']
        
        # Apply dual-layer adaptation
        ai_response = adapt_response(ai_response, immediate_sentiment, conversation_sentiment)
        
        # Append assistant response
        chat_history.append({"role": "assistant", "content": ai_response})

        return jsonify({
            "response": ai_response,
            "immediate_sentiment": immediate_sentiment,
            "conversation_sentiment": conversation_sentiment
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Temporary service issue"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)