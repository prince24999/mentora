from flask import Flask, request, jsonify, render_template
from llama_cpp import Llama

app = Flask(__name__)

# 🧠 Load mô hình GGUF
llm = Llama(model_path="modelQ4.gguf", n_ctx=1024)

# ✨ System prompt định hướng phản hồi
system_prompt = "Bạn là một chuyên gia tâm lý, luôn lắng nghe, thấu hiểu và phản hồi nhẹ nhàng bằng tiếng Việt."

# 🧩 Lưu lịch sử theo IP
user_sessions = {}

@app.route("/")
def home():
    return render_template("index.html")  # ✅ Giao diện chính

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    user_id = request.remote_addr

    if user_id not in user_sessions:
        user_sessions[user_id] = []

    recent_history = user_sessions[user_id][-5:]
    dialogue = "\n".join([f"User: {msg['user']}\nAssistant: {msg['bot']}" for msg in recent_history])
    prompt = f"{system_prompt}\n{dialogue}\nUser: {user_input}\nAssistant:"

    output = llm(
        prompt,
        max_tokens=300,
        temperature=0.7,
        top_k=50,
        top_p=0.6,
        repeat_penalty=1.5,
        frequency_penalty=0.7,
        presence_penalty=0.5,
        stop=["User:", "Assistant:"]
    )

    answer = output["choices"][0]["text"].strip()
    user_sessions[user_id].append({"user": user_input, "bot": answer})

    return jsonify({"response": answer})
