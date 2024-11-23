import os
import shutil
import sys

from flask import Flask, jsonify, request, send_from_directory

from pipeline import run_pipeline

app = Flask(__name__)
UPLOAD_FOLDER = "./raw_data"
RESULT_FOLDER = "./results"
IMAGE_FOLDER = "./output_images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    # Save the file to raw_data folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Run the pipeline
    run_pipeline(file_path)

    with open(os.path.join(RESULT_FOLDER, "results.txt"), "r") as result_file:
        accuracy = result_file.read()

    image_files = os.listdir(IMAGE_FOLDER)

    return jsonify({"accuracy": accuracy, "images": image_files})


@app.route("/images/<path:filename>", methods=["GET"])
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
