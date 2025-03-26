# src/preprocess.py

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import pandas as pd
import os
import re
from transformers import BertTokenizer
import nltk
# Ensure NLTK data is downloaded (best practice: user runs this command separately)
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     print("NLTK 'punkt' not found. Please run: python -m nltk.downloader punkt")
#     # nltk.download('punkt', quiet=True) # Avoid downloading during import
# from nltk.tokenize import word_tokenize

# Import configuration variables
from .config import (
    BERT_MODEL_NAME,
    IMG_SIZE,
    MAX_TOKEN_LEN,
    DATA_FILE,
    IMAGE_DIR,
    IMAGE_FILENAME_COL,
    TEXT_COL,
    SENTIMENT_COL,
    LABEL_MAP,
)

# --- Image Transformations ---
# Define the sequence of transformations for the images
image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resize images to the required input size
    transforms.ToTensor(), # Convert PIL image to PyTorch tensor (scales to [0, 1])
    # Normalize using ImageNet mean and std dev (standard practice for models pre-trained on ImageNet)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Optional: Simple Text Preprocessing ---
# Basic cleaning before tokenization (BERT often handles much of this implicitly)
def preprocess_text_simple(text):
    if not isinstance(text, str):
        text = str(text) # Ensure text is a string
    text = text.lower() # Lowercasing
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\@\w+|\#','', text) # Remove mentions and hashtags (optional)
    # Further cleaning like removing punctuation or stopwords can be added here if needed
    # tokens = word_tokenize(text) # Requires nltk.download('punkt')
    # return " ".join(tokens)
    return text

# --- Custom Dataset Class ---
class MultimodalSentimentDataset(Dataset):
    """
    Custom PyTorch Dataset for loading multimodal sentiment data (text + image).
    Reads data mapping from an Excel file and loads images from structured directories.
    """
    def __init__(self, data_file=DATA_FILE, image_dir=IMAGE_DIR,
                 tokenizer=None, transform=image_transform,
                 max_token_len=MAX_TOKEN_LEN, label_map=LABEL_MAP):
        """
        Args:
            data_file (str): Path to the Excel file (.xlsx) containing text, image filenames, and sentiments.
            image_dir (str): Path to the base directory containing sentiment subfolders (Positive, Negative, Neutral).
            tokenizer: Pre-initialized BERT tokenizer instance.
            transform (callable, optional): Transformations to apply to the images.
            max_token_len (int): Maximum length for tokenized text sequences.
            label_map (dict): Dictionary mapping sentiment strings to integer labels.
        """
        self.image_dir = image_dir
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(BERT_MODEL_NAME) # Initialize if not provided
        self.transform = transform
        self.max_token_len = max_token_len
        self.label_map = label_map

        # Load data mapping from Excel
        try:
            self.data_df = pd.read_excel(data_file)
            print(f"Loaded data mapping from {data_file}. Found {len(self.data_df)} entries.")
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_file}.")
            print("Please ensure you have downloaded the dataset and placed 'LabeledText.xlsx' in the project root.")
            self.data_df = pd.DataFrame() # Create empty dataframe to avoid errors later
            return # Stop initialization if file not found

        # --- Validate required columns ---
        required_cols = [IMAGE_FILENAME_COL, TEXT_COL, SENTIMENT_COL]
        if not all(col in self.data_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.data_df.columns]
            print(f"Error: Missing required columns in {data_file}: {missing}")
            print(f"Expected columns based on config: {required_cols}")
            self.data_df = pd.DataFrame() # Make dataframe empty
            return

        # --- Preprocessing and Path Construction ---
        # Apply basic text cleaning (optional)
        # self.data_df['cleaned_text'] = self.data_df[TEXT_COL].apply(preprocess_text_simple)
        # Use original text if not applying cleaning
        self.data_df['cleaned_text'] = self.data_df[TEXT_COL].astype(str)

        # Map sentiment strings to integer labels
        self.data_df['label'] = self.data_df[SENTIMENT_COL].map(self.label_map)

        # Drop rows where sentiment mapping failed (resulted in NaN)
        initial_len = len(self.data_df)
        self.data_df.dropna(subset=['label'], inplace=True)
        if len(self.data_df) < initial_len:
            print(f"Warning: Dropped {initial_len - len(self.data_df)} rows due to invalid sentiment labels.")

        # Ensure labels are integer type
        self.data_df['label'] = self.data_df['label'].astype(int)

        # Construct full image path based on sentiment and filename
        # Assumes image_dir contains subfolders named EXACTLY like the keys in INV_LABEL_MAP (Positive, Negative, Neutral)
        inv_label_map = {v: k for k, v in self.label_map.items()} # Create reverse map {0: 'Positive', ...}
        def get_image_path(row):
            sentiment_folder = inv_label_map.get(row['label'])
            if sentiment_folder:
                # Strip potential leading/trailing whitespace from filename
                filename = str(row[IMAGE_FILENAME_COL]).strip()
                return os.path.join(self.image_dir, sentiment_folder, filename)
            return None # Return None if label/sentiment folder is invalid

        self.data_df['full_image_path'] = self.data_df.apply(get_image_path, axis=1)

        # Report rows with missing image paths (e.g., due to bad labels)
        missing_paths = self.data_df['full_image_path'].isnull().sum()
        if missing_paths > 0:
             print(f"Warning: Could not construct image paths for {missing_paths} entries (check sentiment labels and filenames).")
             self.data_df.dropna(subset=['full_image_path'], inplace=True) # Drop rows where path construction failed

        print(f"Dataset ready. Final number of samples: {len(self.data_df)}")


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data_df)

    def __getitem__(self, index):
        """
        Fetches the sample at the given index, preprocesses it, and returns it.

        Args:
            index (int): The index of the sample to fetch.

        Returns:
            dict: A dictionary containing:
                'input_ids': Token IDs for the text.
                'attention_mask': Attention mask for the text.
                'image': Preprocessed image tensor.
                'label': Integer sentiment label.
        """
        # Get the data row for the given index
        data_row = self.data_df.iloc[index]

        # --- Text Processing ---
        text = data_row['cleaned_text']
        try:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,       # Add '[CLS]' and '[SEP]'
                max_length=self.max_token_len, # Pad & truncate to max length
                return_token_type_ids=False,   # Not needed for basic BERT classification
                padding='max_length',          # Pad to max_length
                truncation=True,               # Truncate to max_length
                return_attention_mask=True,    # Return attention mask
                return_tensors='pt',           # Return PyTorch tensors
            )
            input_ids = encoding['input_ids'].flatten()
            attention_mask = encoding['attention_mask'].flatten()
        except Exception as e:
             print(f"Error tokenizing text at index {index}: '{text}'. Error: {e}")
             # Return dummy tensors if tokenization fails
             input_ids = torch.zeros(self.max_token_len, dtype=torch.long)
             attention_mask = torch.zeros(self.max_token_len, dtype=torch.long)


        # --- Image Processing ---
        img_path = data_row['full_image_path']
        try:
            # Open image, ensure it's RGB (handle grayscale, RGBA etc.)
            image = Image.open(img_path).convert('RGB')
            # Apply transformations
            image = self.transform(image)
        except FileNotFoundError:
            print(f"Warning: Image file not found at {img_path} (index {index}). Using zero tensor.")
            image = torch.zeros((3, IMG_SIZE, IMG_SIZE)) # Return a zero tensor
        except UnidentifiedImageError:
            print(f"Warning: Cannot identify image file (corrupted?) at {img_path} (index {index}). Using zero tensor.")
            image = torch.zeros((3, IMG_SIZE, IMG_SIZE))
        except Exception as e:
            print(f"Warning: Generic error processing image {img_path} (index {index}): {e}. Using zero tensor.")
            image = torch.zeros((3, IMG_SIZE, IMG_SIZE))

        # --- Label ---
        label = torch.tensor(data_row['label'], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image,
            'label': label
        }

# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    # This block runs only when the script is executed directly (e.g., python src/preprocess.py)
    print("Testing MultimodalSentimentDataset...")

    # Check if data files exist before attempting to load
    if not os.path.exists(DATA_FILE):
         print(f"Error: Cannot run test. Data file '{DATA_FILE}' not found.")
         print("Please download the dataset and place it in the project root.")
    elif not os.path.exists(IMAGE_DIR):
         print(f"Error: Cannot run test. Image directory '{IMAGE_DIR}' not found.")
         print("Please download the dataset and place it in the project root.")
    else:
        # Initialize tokenizer (needed for the test)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

        # Create dataset instance
        dataset = MultimodalSentimentDataset(tokenizer=tokenizer)

        # Check dataset length and try fetching the first item
        if len(dataset) > 0:
            print(f"Dataset loaded successfully with {len(dataset)} samples.")
            first_item = dataset[0]
            print("\nSample data from the first item:")
            print("Keys:", first_item.keys())
            print("Input IDs shape:", first_item['input_ids'].shape)
            print("Attention Mask shape:", first_item['attention_mask'].shape)
            print("Image shape:", first_item['image'].shape)
            print("Label:", first_item['label'])

            # Try fetching another item
            if len(dataset) > 1:
                 second_item = dataset[1]
                 print("\nSuccessfully fetched second item.")
            else:
                 print("\nDataset has only one item.")

        else:
            print("Dataset is empty. Check for errors during initialization.")
