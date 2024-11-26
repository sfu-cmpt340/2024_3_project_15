import os

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from flask_cors import CORS

from pipeline import run_pipeline

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "./raw_data"
RESULT_FOLDER = "./results"
IMAGE_FOLDER = "./images"
CLEANED_DATA_FOLDER = "./cleaned_data"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "files" not in request.files:
            return jsonify({"error": "No file part"}), 400

        files = request.files.getlist("files")

        if not files or files[0].filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Clear previous files in the folders before processing new files
        for folder in [UPLOAD_FOLDER, RESULT_FOLDER, IMAGE_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

        # Save and process each file
        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

        run_pipeline()

        # Fetch files from the images folder
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".png")]
        text_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".txt")]

        if not image_files or not text_files:
            return jsonify({"error": "No results found in images folder"}), 500

        image_urls = [
            url_for("serve_file", filename=image, _external=True)
            for image in image_files
        ]

        return jsonify({"images": image_urls, "text_files": text_files}), 200
    except Exception as e:
        print("Error in /upload:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500


@app.route("/files/<path:filename>", methods=["GET"])
def serve_file(filename):
    file_path = os.path.join(IMAGE_FOLDER, filename)
    if not os.path.exists(file_path):
        return "File not found", 404

    response = send_from_directory(IMAGE_FOLDER, filename)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/clear", methods=["POST"])
def clear_results():
    try:

        folders_to_clear = [
            UPLOAD_FOLDER,
            RESULT_FOLDER,
            IMAGE_FOLDER,
            CLEANED_DATA_FOLDER,
        ]

        for folder in folders_to_clear:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

        extracted_features_path = os.path.join(os.getcwd(), "extracted_features.csv")
        if os.path.exists(extracted_features_path):
            try:
                os.unlink(extracted_features_path)
                print(f"Deleted {extracted_features_path}")
            except Exception as e:
                print(f"Error deleting {extracted_features_path}: {e}")

        return jsonify({"status": "success"}), 200
    except Exception as e:
        print("Error in /clear:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
