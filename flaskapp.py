from flask import Flask,render_template,request,jsonify
from chat import chatfun

app = Flask(__name__)

@app.get("/")
def home():
    return render_template('index.html')

@app.post("/pred")
def pred():
    text = request.get_json().get("message")
    reply = chatfun(text)
    message = {"answer":reply}
    return jsonify(message)

if __name__=='__main__':
    app.run(host='0.0.0.0')