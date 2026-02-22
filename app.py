from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss, pickle, numpy as np, os
from groq import Groq

app = Flask(__name__)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load index and chunks if they exist
index = None
chunks = []

if os.path.exists("curriculum.index"):
    index = faiss.read_index("curriculum.index")

if os.path.exists("chunks.pkl"):
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

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

    context = ""

    if index is not None and chunks:
        q_vector = model.encode([question])
        _, indices = index.search(np.array(q_vector), k=5)

        relevant = []
        for i in indices[0]:
            chunk = chunks[i]
            if (level.lower() in chunk["level"].lower() and
                subject.lower() in chunk["subject"].lower()):
                relevant.append(chunk["content"])

        context = "\n\n".join(relevant[:3])

    prompt = f"""You are a helpful Tanzania curriculum tutor for Form I to VI students.
{"Use the following curriculum content to answer:" if context else "Answer based on your knowledge."}

{context}

Question: {question}

Answer clearly and step by step:"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

