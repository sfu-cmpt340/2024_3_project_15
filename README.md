# SFU CMPT 340 Project - EEG Scrolling Direction Classification with Muse2 Brain Sensing Headband 
This project focuses on collecting, cleaning, feature extracting, and classifying EEG data from the Muse2 Brain Sensing Headband. The goal is to classify the direction of scrolling (up or down) based on brain activity using machine learning models like Random Forest, 


## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EYcEhogX3nlMlobLCvc9I1UBQAROq3b5g4AKcHswM16LWg?e=0jHbXh) | [Slack channel](https://app.slack.com/client/T07K7SWL5A4/C07JKF7EBML) | [Project report](https://www.overleaf.com/project/66d0b103964b3acdf17669aa) |
|-----------|---------------|-------------------------|

## Video/demo/GIF
PUT GIF HERE

## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Troubleshooting](#troubleshoot)


<a name="demo"></a>
## 1. Demo - Running Website and Visualizing EEG Data

### Running the website 

1. Navigate to the `EEG_Web` directory:
     ```bash
     cd src/EEG_Web
     ```
2. Run the `server.py` script:
     ```bash
     python server.py
     ```
3. Open the website URL from the terminal

4. Upload a raw EEG csv file and you should see something similar to below

![pic1](https://github.com/user-attachments/assets/2e9319b5-b383-4f6d-8efa-c1284084a479)


### Visualizing EEG Data

1. **Scrolling EEG Data Visualization**:
   - Navigate to the `src > EEG_Data > dataProcessing > dataAnalyze` directory
     ```bash
     cd .\src\EEG_Data\dataProcessing\dataAnalyze\
     ```
      
   - Run the `visualize_scrolling_eeg.py` or `visualize_swiping_eeg.py`  script:
     ```bash
     python visualize_scrolling_eeg.py
     ```   
     ```bash
     python visualize_swiping_eeg.py
     ```
   - This script will generate EEG signal plots for scrolling/swiping data, giving an insight into how EEG activity varies during these interactions.

   ![pic2](https://github.com/user-attachments/assets/868b579c-8489-49d3-9615-650dc0bf5622)
   <br>
   ![pic3](https://github.com/user-attachments/assets/90d4c57e-a518-4e4b-b9e6-75f8a061c0c8)

### Directory Overview

To make it easier to understand the project structure, here's a brief summary:

```bash
repository
├── src                          ## Root folder
  ├── EEG_Data                   ## EEG data and process
    ├── dataProcessing           ## Scripts for processing (visualization, cleaning, extracting)
    ├── MLModel                  ## Different iterations of the ML models 
  ├── EEG_Web                    ## Code for the website 
├── README.md                    
├── requirements.yml             
```

<a name="installation"></a>
## 2. Installation

To run this project, set up your development environment by following the steps below:

### Prerequisites:
- Ensure you have at least **Python 3.10** 
- Install a package manager (`pip`).

### Using `pip`:
1. **Clone the Repository**:
    ```bash
    git clone git@github.com:sfu-cmpt340/2024_3_project_15.git
    ```
2. **Create a virtual environment**:
    ```bash
    python -m venv .venv
    ```
3. **Activate the virtual environment**:

   Windows: `.\.venv\Scripts\activate`<br>
   Mac: `source ./venv/bin/activate`
4. **Install Dependencies**:
    ```bash
    pip install -r requirements.yml
    ```

   **Dependencies include**:
   - `matplotlib==3.9.2`
   - `pandas==2.2.3`
   - `scipy==1.14.1`
   - `mne==1.8.0`
   - `scikit-learn`
   - `flask==3.1.0`
   - `Flask-Cors==5.0.0`

### Verifying the Setup:
1. Test the installation by running:
    ```bash
    python -c "import matplotlib, pandas, scipy, mne, sklearn, flask; print('All dependencies are installed.')"
    ```

This ensures the environment is correctly set up and ready to run the project.


<a name="repro"></a>
## 3. Reproduction

Follow these steps to reproduce the results of this project:

### Step 1: Dataset Preparation
1. **Collect EEG Data**:
   - Download and install the MuseLab 1.9.5 onto a MAC. https://drive.google.com/drive/folders/1oy0haqORt55Lk_oW3Gn6t15Zwi0a6cpf
   - Download Mobile App: Muse:Brain Health & Sleep App from App Store.
   - Connect Muse2 with Mobile App.
   - Use the Muse Headband and the Muse app to record around 20 seconds of scrolling up or scrolling down EEG signals (See entire demo for detailed setup steps).
   - Save the recordings as a CSV file (Make sure to include 'up' or 'down' in the csv filename)
   
2. **Preprocess the Data**:
   - Navigate to the src/EEG_Web directory:
      ```bash
      cd src/EEG_Web
      ```
   - Run the server.py script to start the web application:
      ```bash
      python server.py
      ```
   - This will start a website that runs the EEG data processing, analyzing, and classification.
   - Click the link in the terminal to open the web page.

3. **Upload Data Set**:
   - Click the 'Upload and Analyze' button to classify the CSV data

4. **Result**:
   - The processed dataset will be analyzed and classified with the confusion matrix, ROC curve, and precision recall curve on the web page.

Notes: 
- Instead of manually collecting data with the museband, you can also test by uploading the data in `EEG_Data > dataProcessing > EEG_Data > museFiles_scrolling > original_data`
- The ROC curve and precision recall curve only displays if both scrolling up and scrolling down data is uploaded 

<a name="troubleshoot"></a>
## 4. Troubleshooting
- If you get the `Error occured while processing the EEG signal` error message on the website, make sure the scipy version is 1.14.1 (If you can't install scipy 1.14.1, make sure your Python version is up to date)