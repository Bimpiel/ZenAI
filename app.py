from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import chromadb
import openai
from dotenv import load_dotenv
import os
from transformers import pipeline  # For emotion classification
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Flask setup
app = Flask(__name__)
CORS(app)

# Database setup - using smaller collection if possible
CHROMA_PATH = r"chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="mental_docs",
    metadata={"hnsw:space": "cosine"}  # More memory-efficient
)

# Load a lightweight emotion classifier
try:
    emotion_classifier = pipeline(
        "text-classification", 
        model="finiteautomata/bertweet-base-emotion-analysis",
        framework="pt",
        device=-1  # Use CPU to save memory
    )
except Exception as e:
    print(f"Couldn't load emotion classifier: {e}")
    emotion_classifier = None

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Retrieve relevant data once per session - limit to essential documents
retrieved_data = collection.query(
    query_texts=["general mental health support"], 
    n_results=3  # Reduced from 5 to save memory
)
retrieved_data = retrieved_data.get("documents", ["No relevant data found."])

# Global chat history with initial system prompt
chat_history = [
    {
        "role": "system", 
        "content": f"""
        You are a mental health chatbot offering support. Show empathy and use the retrieved data for accuracy. 
        Only answer with factual statements from the retrieved data.
        
        Guidelines:
        1. Only answer mental health-related questions
        2. Keep responses short (1-2 sentences max)
        3. Use a calm, friendly tone with occasional emojis 😊
        4. Avoid generic sympathy statements
        5. Adapt responses based on the user's detected emotion
        
        Retrieved data: {retrieved_data}
        """
    }
]

# Maximum chat history length
MAX_CHAT_HISTORY_LENGTH = 10  # Reduced from 15 to save memory

def classify_emotion(text):
    """Classify emotion from text using a lightweight model"""
    if not emotion_classifier:
        return "neutral"
    
    try:
        result = emotion_classifier(text[:512])  # Limit input size
        return result[0]['label'].lower()
    except Exception as e:
        print(f"Emotion classification error: {e}")
        return "neutral"

def adapt_response_based_on_emotion(response, emotion):
    """Lightweight adaptation of response based on emotion"""
    if emotion == "joy":
        return response + " 😊 It's great to hear you're feeling positive!"
    elif emotion == "anger":
        return response + " I sense you might be frustrated. Let's work through this together."
    elif emotion == "sadness":
        return response + " 🫂 I'm here to listen if you want to share more."
    elif emotion == "fear":
        return response + " It's okay to feel this way. You're safe here."
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

    # Classify user emotion (lightweight operation)
    emotion = classify_emotion(user_input)
    
    # Append user message with emotion context
    chat_history.append({
        "role": "user", 
        "content": f"[{emotion}] {user_input}"  # Prefix emotion for context
    })

    # Trim history if needed (keep system message)
    if len(chat_history) > MAX_CHAT_HISTORY_LENGTH:
        chat_history = [chat_history[0]] + chat_history[-(MAX_CHAT_HISTORY_LENGTH-1):]

    try:
        # Use a more memory-efficient model
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # More memory-efficient than gpt-4
            messages=chat_history,
            max_tokens=100,  # Limit response length
            temperature=0.7  # Balance creativity and focus
        )

        ai_response = response['choices'][0]['message']['content']
        
        # Lightweight emotion adaptation
        ai_response = adapt_response_based_on_emotion(ai_response, emotion)
        
        # Append assistant response
        chat_history.append({"role": "assistant", "content": ai_response})

        return jsonify({
            "response": ai_response,
            "detected_emotion": emotion  # Optional: send back to frontend
        })

    except Exception as e:
        print(f"Error in chat: {e}")
        return jsonify({"error": "Sorry, I'm having trouble responding right now"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)  # debug=False for production