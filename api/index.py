from flask import Flask, send_from_directory
import subprocess
import os

app = Flask(__name__)

@app.route("/")
def home():
    return send_from_directory('../', 'streamlit_app.py')

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory('../', path)

@app.route("/healthz")
def healthz():
    return "OK"

if __name__ == "__main__":
    if not os.path.exists("public"):
        os.makedirs("public")
    subprocess.Popen(["streamlit", "run", "../streamlit_app.py", "--browser.serverAddress", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"])
    app.run(host="0.0.0.0", port=3000)
