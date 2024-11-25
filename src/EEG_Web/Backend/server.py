import os
import shutil
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from pipeline import run_pipeline

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "./raw_data"
RESULT_FOLDER = "./results"
IMAGE_FOLDER = "./images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        run_pipeline(file_path)

        # Fetch files from the images folder
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".png")]
        text_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".txt")]

        if not image_files or not text_files:
            return jsonify({"error": "No results found in images folder"}), 500

        return jsonify({"images": image_files, "text_files": text_files}), 200
    except Exception as e:
        print("Error in /upload:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500


@app.route("/files/<path:filename>", methods=["GET"])
def serve_file(filename):
    file_path = os.path.join(IMAGE_FOLDER, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_from_directory(IMAGE_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)