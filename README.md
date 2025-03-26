# Multimodal Sentiment Analysis (Text & Image)

This project implements a multimodal sentiment analysis model using text captions and corresponding images from a Twitter dataset. It utilizes BERT for text feature extraction and ResNet for image feature extraction, fusing them for a final sentiment classification (Positive, Negative, Neutral).

## Dataset (Download Required)

This repository **does not** include the dataset. You must download it from Kaggle:

*   **Dataset Link:** [https://www.kaggle.com/datasets/dunyajasim/twitter-dataset-for-sentiment-analysis?resource=download](https://www.kaggle.com/datasets/dunyajasim/twitter-dataset-for-sentiment-analysis?resource=download)

**Download Methods:**

1.  **Manual Download (Easier):**
    *   Click the link above.
    *   Log in to your Kaggle account (required).
    *   Click the "Download" button on the Kaggle page. This will download a `.zip` file (e.g., `archive.zip`).
    *   **Extract** the contents of the `.zip` file directly into the **root directory** of this project (where this README file is).
2.  **Kaggle API (Advanced):**
    *   Ensure you have the Kaggle API installed (`pip install kaggle`).
    *   Set up your Kaggle API credentials (download `kaggle.json` from your Kaggle account settings and place it in `~/.kaggle/kaggle.json` or `C:\Users\<Windows-username>\.kaggle\kaggle.json`).
    *   Navigate to the root directory of this project in your terminal.
    *   Run the download command:
        ```bash
        kaggle datasets download -d dunyajasim/twitter-dataset-for-sentiment-analysis -p . --unzip
        ```
        *(This downloads the dataset to the current directory (`-p .`) and automatically unzips (`--unzip`)).*

**Expected Data Structure (after download & extraction):**

After downloading and extracting, your project's root directory should contain:

*   `LabeledText.xlsx`
*   `Images/` (a directory)
    *   `Images/Positive/`
    *   `Images/Negative/`
    *   `Images/Neutral/`
*   `src/` (and other project files)

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/HonzaOndrasek/OPIT_NLP-and-Computer-Vision.git
    cd OPIT_NLP-and-Computer-Vision
    ```
2.  **Create Environment & Install Dependencies:** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Download Dataset:** Follow the instructions in the "Dataset (Download Required)" section above to download and place `LabeledText.xlsx` and the `Images/` folder in the project root.
4.  **Download NLTK Data:**
    ```bash
    python -m nltk.downloader stopwords punkt
    ```

## Usage

To train and evaluate the multimodal model:
```bash
python src/train.py
