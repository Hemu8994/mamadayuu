from flask import Flask, request, jsonify
import os
from groq import Groq

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Mamadayuu AI is running!"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    level = data.get("level", "")
    subject = data.get("subject", "")

    prompt = f"""You are a helpful Tanzania curriculum tutor for Form I to VI students.
You follow the Tanzania Institute of Education (TIE) curriculum.

Student Level: {level}
Subject: {subject}
Question: {question}

Answer clearly, step by step, in simple English suitable for a Tanzania secondary school student:"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    return jsonify({"answer": answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
