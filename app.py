from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import chromadb
import openai
from dotenv import load_dotenv
import os
from transformers import pipeline

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

# Emotion classification setup
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Retrieve relevant data once per session
retrieved_data = collection.query(query_texts=["general mental health support"], n_results=5)
retrieved_data = retrieved_data.get("documents", ["No relevant data found."])

# Global chat history
chat_history = [
    {"role": "system", "content": f"""
        You are a mental health chatbot offering support. Show empathy and use the retrieved data for accuracy. 
        Only answer with factual statements from the retrieved data.
     
        All answer should be related to mental health. Meaning that if someone ask about any unrelated topic that is not related to any provided data, please do not answer!
        Although you are an AI chatbot, your answers must feel human! For example, your tone should be calm, trustworthy, and friendly. Your language should be rather informal.
        Maybe add some humor to elevate the mood of the user. Potentially answer with emojis to make it feel real!
        
        Keep in mind that the person using this doesn't necessarily want to talk to anyone in real life. They might feel scared or overwhelmed over the thought of sharing their feeling.
        As a result, they are coming to you for advice!

        Try short answers. Not too long of a paragraph.
        For example, Input: Can you help me manage my stress level? Output: of course, what has been bothering you?
     
        Avoid the generic: I’m really sorry to hear that or anything related to that.
        Retrieved data: {retrieved_data}
    """}
]

# Set a maximum chat history length
MAX_CHAT_HISTORY_LENGTH = 15  # <-- for example, keep last 15 messages

# Emotion detection function
def detect_emotion(text):
    result = emotion_classifier(text)
    emotion = result[0]['label']
    return emotion

# Adjust the response based on emotion
def adjust_response_for_emotion(response, emotion):
    if emotion == 'sadness':
        return response + "\nIt’s tough, but you're not alone in this. I’m here to help you through it 💙"
    elif emotion == 'anger':
        return response + "\nI understand that you're upset. Let's try to work through this together."
    elif emotion == 'fear':
        return response + "\nI can sense you're anxious. Take a deep breath, and let's go step by step."
    elif emotion == 'joy':
        return response + "\nIt’s great to hear that you're feeling good! Keep that positive energy going ✨"
    else:
        return response

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

    # Detect the emotion in the user's message
    emotion = detect_emotion(user_input)
    
    # Append the emotion to chat history (optional for better context)
    chat_history.append({"role": "user", "content": f"[{emotion}] {user_input}"})

    # Keep chat history within the limit
    if len(chat_history) > MAX_CHAT_HISTORY_LENGTH:
        chat_history = [chat_history[0]] + chat_history[-(MAX_CHAT_HISTORY_LENGTH-1):]

    # Use OpenAI's ChatCompletion API to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # You can change this to another model like gpt-3.5-turbo
        messages=chat_history,  # Provide the chat history to maintain conversation context
    )

    # Extract the assistant's response from the API response
    ai_response = response['choices'][0]['message']['content']

    # Adjust the AI response based on the detected emotion
    ai_response = adjust_response_for_emotion(ai_response, emotion)

    # Append the assistant's response to the chat history
    chat_history.append({"role": "assistant", "content": ai_response})

    # Keep chat history within the limit again
    if len(chat_history) > MAX_CHAT_HISTORY_LENGTH:
        chat_history = [chat_history[0]] + chat_history[-(MAX_CHAT_HISTORY_LENGTH-1):]

    # Return the response in JSON format
    return jsonify({"response": ai_response, "emotion": emotion})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to port 5000 if no PORT environment variable is set
    app.run(debug=True, host="0.0.0.0", port=port)
