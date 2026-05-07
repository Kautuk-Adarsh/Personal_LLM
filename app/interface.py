# app/interface.py
import os
from flask import Flask, request, jsonify, render_template
from app.chat import ChatEngine


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
engine = None


def create_app(model_path: str, tokenizer_path: str):
    global engine
    engine = ChatEngine(model_path, tokenizer_path)
    return app


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    print("CHAT ROUTE HIT")
    data = request.get_json()
    print(f"Data received: {data}")
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please ask a question."})
    if engine is None:
        return jsonify({"answer": "Engine not initialized."})
    answer = engine.answer(question)
    print(f"Answer: {answer}")
    return jsonify({"answer": answer})
