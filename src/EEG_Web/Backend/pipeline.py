import os

from clean_data import clean
from feature_extraction import extract_features
from voting_ensemble import classify_with_models


def run_pipeline(test_data_path):
    # Initialize folder paths
    RAW_DATA_FOLDER = "./raw_data"
    CLEANED_DATA_FOLDER = "./cleaned_data"
    RESULT_FOLDER = "./results"
    IMAGE_FOLDER = "./images"

    # Ensure necessary folders exist
    os.makedirs(CLEANED_DATA_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    clean(RAW_DATA_FOLDER)

    extract_features(CLEANED_DATA_FOLDER, "extracted_features.csv")

    classify_with_models("extracted_features.csv", svm=0)

    # Save the results
    with open(os.path.join(RESULT_FOLDER, "results.txt"), "w") as result_file:
        result_file.write("Model accuracy and other metrics")
