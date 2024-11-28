# SFU CMPT 340 Project - EEG Action Classification with Muse2 Brain Sensing Headband 
This project focuses on collecting, cleaning, and analyzing EEG data from the Muse2 Brain Sensing Headband. The goal is to classify user actions (e.g., scrolling or swiping) based on brain activity using machine learning models.


## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EYcEhogX3nlMlobLCvc9I1UBQAROq3b5g4AKcHswM16LWg?e=0jHbXh) | [Slack channel](https://app.slack.com/client/T07K7SWL5A4/C07JKF7EBML) | [Project report](https://www.overleaf.com/project/66d0b103964b3acdf17669aa) |
|-----------|---------------|-------------------------|


- Timesheet: Link your timesheet (pinned in your project's Slack channel) where you track per student the time and tasks completed/participated for this project/
- Slack channel: Link your private Slack project channel.
- Project report: Link your Overleaf project report document.


## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.

[Demo Video](https://drive.google.com/file/d/1GscGs1FrE-tUDiTAucsPe4q-xS4oyQJT/view?usp=sharing)

## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example Demo

This section demonstrates how to run the web and upload the .

### Start the Web server

1. **Run Pipeline for cleaning data**:
   - Navigate to the `EEG_Web` directory:
     ```bash
     cd src/EEG_Web
     ```
   - Run the `server.py` script:
     ```bash
     python server.py
     ```
     
2. **Upload Data and Analyze**:
   - Upload data sets on the webpage and wait for analyze

### Visualizing EEG Data

1. **Scrolling EEG Data Visualization**:
   - Navigate to the `visualizations` directory:
     ```bash
     cd ../visualizations
     ```
   - Run the `visualize_scrolling_eeg.py` script:
     ```bash
     python visualize_scrolling_eeg.py
     ```
   - This script will generate EEG signal plots for scrolling actions, giving an insight into how EEG activity varies during these interactions.

2. **Visualization the result**:
   - After analyze, web will output the visualization for the different models result.

### Directory Overview

To make it easier to understand the project structure, here's a brief summary:

```bash
repository
├── src                          ## source code of the package itself
  ├── EEG_Data                   ## EEG data and process
  ├── EEG_Web                   ## If needed, documentation   
├── README.md                    ## You are here
├── requirements.yml             ## If you use conda
```

<a name="installation"></a>
## 2. Installation

To run this project, set up your development environment by following the steps below:

### Prerequisites:
- Ensure you have **Python 3.10** or a compatible version installed.
- Install a package manager (`pip`).

### Using `pip`:
1. **Clone the Repository**:
    ```bash
    git clone git@github.com:sfu-cmpt340/2024_3_project_15.git
    ```
2. **Navigate to the Project Directory**:
    ```bash
    cd EEGMuse
    ```
3. **Install Dependencies**:
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

### Notes:
- Ensure you are using Python 3.10 or a compatible version.
- If additional packages are required, include them in the `requirements.yml` or `requirements.txt` file for consistency.

<a name="repro"></a>
## 3. Reproduction

Follow these steps to reproduce the results of this project:

### Step 1: Dataset Preparation
1. **Collect EEG Data**:
   - Download and install the MuseLab to OS. https://drive.google.com/drive/folders/1oy0haqORt55Lk_oW3Gn6t15Zwi0a6cpf
   - Download Mobile App: Muse:Brain Health & Sleep App from App Store.
   - Connect Muse2 with Mobile App.
   - Use the Muse2 Brain Sensing Headband and the Muse Direct app to record raw EEG signals.
   - Save the recordings as CSV files for preprocessing.

2. **Preprocess the Data**:
   - Navigate to the `src/EEG_Web` directory:
     ```bash
     cd src/EEG_Web
     ```
   - Run the `server.py` script to start the web application:
     ```bash
     python server.py
     ```
   - This will initiate a web application that running the EEG data processing, analyzing.
   - Click the link in the termianl to open the web page.

3. **Upload Data Set**:
   - Click 'Upload and Analyze' button to upload the recording CSV data and wait:
     
4. **Result**:
   - The processed dataset will be analyzed and output the results on the web page.

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
