"""
Take in a test_data.csv and then cleans it -> feature extraction -> classifies/labels it -> Generate the Ethan


"""

from clean_data import clean
from feature_extraction import extract_features
from voting_ensemble import classify_with_models

if __name__ == "__main__":

    # Initialize folder paths
    RAW_DATA_FOLDER = "./raw_data"
    CLEANED_DATA_FOLDER = "./cleaned_data"

    # 1. Clean all the data found in raw_data folder and save into cleaned_data folder
    clean(RAW_DATA_FOLDER)

    # 2. Feature extraction on the cleaned data and save into extracted_features.csv
    extract_features(CLEANED_DATA_FOLDER, "extracted_features.csv")

    # 3. Classify the data and generate Ethan
    classify_with_models("extracted_features.csv", svm=0)
