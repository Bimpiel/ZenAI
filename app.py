from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Flask setup
app = Flask(__name__)
CORS(app)

# Database setup
CHROMA_PATH = r"chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="mental_docs")

# OpenAI setup
client = OpenAI()

# Retrieve relevant data once per session
retrieved_data = collection.query(query_texts=["general mental health support"], n_results=5)
retrieved_data = retrieved_data.get("documents", ["No relevant data found."])

# Global chat history
chat_history = [
    {"role": "system", "content": f"""
        You are a mental health chatbot offering support. Show empathy and use the retrieved data for accuracy. 
        Only answer with factual statements from the retrieved data.
     
        All answer should be related to mental health. Meaning that if someone ask about any unrelated topic that is not related to any provided data, please do not answer!
        Although you are not answering, reply with I don't know much rather than being, i cant talk about this topic!
        While this is an AI chatbot, you answers must feel humaine! For example: your tone should be calm, trustworthy and friendly. Your language should be rather informal.
        Maybe add some humor to elevate the mood of the user. Potentially answer with emojis to make it feel real!
        
        Keep in mind that the person using this doesn't necessarily want to talk to anyone in real life. They might feel scared or overwhelmed over the thought of sharing their feeling.
        As a result, they are coming to you for advice!

        Try short answers. Not too long of a paragraph.
        For example, Input: Can you help me manage my stress level? Output: of course, what has been bothering you?
     
        Avoid the generic: I’m really sorry to hear that or anything related to that.
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_history,
    )

    ai_response = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": ai_response})

    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(debug=True)
