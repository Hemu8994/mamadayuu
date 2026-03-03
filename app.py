from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this for CORS support
import os
import json
import re
from groq import Groq

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Mamadayuu AI is running!"})

# Original endpoint
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

# NEW: Endpoint that matches what your JavaScript expects
@app.route("/ai-edu-generator/api/book_rag.php", methods=["POST"])
def book_rag():
    try:
        # Get data from form-data (since JavaScript sends FormData)
        question = request.form.get('question', '')
        book_meta = request.form.get('book_meta', '')
        history_json = request.form.get('history', '[]')
        csrf_token = request.form.get('csrf_token', '')
        
        # Parse history
        try:
            history = json.loads(history_json)
        except:
            history = []
        
        # Extract level and subject from book_meta
        level = extract_level(book_meta)
        subject = extract_subject(book_meta)
        book_title = extract_title(book_meta)
        
        # Build conversation context from history
        conversation_context = ""
        if history and len(history) > 0:
            conversation_context = "Previous conversation:\n"
            for msg in history[-4:]:  # Last 4 messages for context
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                conversation_context += f"{role.capitalize()}: {content}\n"
        
        # Create prompt for the AI
        prompt = f"""You are a helpful Tanzania curriculum tutor specializing in the book:

{book_meta}

{conversation_context}
Current question: {question}

Instructions:
1. Answer based on the book content above
2. If the question is not related to this book, politely redirect to the book's topics
3. Use simple language suitable for Tanzanian secondary students
4. Provide examples where helpful
5. If you don't know something based on the book, say so honestly

Answer:"""
        
        # Call Groq API
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # You can use a larger model if needed
            messages=[
                {"role": "system", "content": "You are a Tanzania Institute of Education (TIE) curriculum expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        return jsonify({
            "success": True,
            "reply": answer
        })
        
    except Exception as e:
        print(f"Error in book_rag: {str(e)}")  # This will show in Render logs
        return jsonify({
            "success": False,
            "error": "Failed to get AI response. Please try again."
        }), 500

# Helper functions to parse book_meta
def extract_level(meta):
    """Extract education level from book meta"""
    # Look for class/level information
    class_match = re.search(r'Class:\s*(.*?)(?:\n|$)', meta)
    if class_match:
        class_name = class_match.group(1)
        if 'Form' in class_name or 'Form' in meta:
            return 'Secondary'
        elif 'Primary' in class_name or 'Primary' in meta:
            return 'Primary'
        elif 'A-Level' in class_name or 'A-Level' in meta:
            return 'A-Level'
    return 'Secondary'  # Default

def extract_subject(meta):
    """Extract subject from book meta"""
    subject_match = re.search(r'Subject:\s*(.*?)(?:\n|$)', meta)
    if subject_match:
        return subject_match.group(1).strip()
    return 'General'

def extract_title(meta):
    """Extract book title from meta"""
    title_match = re.search(r'Title:\s*(.*?)(?:\n|$)', meta)
    if title_match:
        return title_match.group(1).strip()
    return 'the book'

# Add a simple test endpoint
@app.route("/test-rag", methods=["GET"])
def test_rag():
    return jsonify({
        "message": "RAG endpoint is working!",
        "endpoints": {
            "book_rag": "/ai-edu-generator/api/book_rag.php",
            "ask": "/ask"
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
