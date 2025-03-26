# src/config.py

import torch

# --- Data Configuration ---
# Paths assume 'LabeledText.xlsx' and 'Images/' are in the project root directory
DATA_FILE = "LabeledText.xlsx"
IMAGE_DIR = "Images" # Base directory containing Positive/Negative/Neutral subfolders

# Column names expected in the Excel file (Adjust if your column names are different)
IMAGE_FILENAME_COL = "File Name"
TEXT_COL = "Caption"
SENTIMENT_COL = "Sentiment" # Assumes values 'Positive', 'Negative', 'Neutral'

# --- Model Configuration ---
BERT_MODEL_NAME = 'bert-base-uncased'
IMG_SIZE = 224 # Standard input size for ResNet
NUM_CLASSES = 3 # Positive, Negative, Neutral

# Feature sizes (depend on the models chosen)
BERT_OUTPUT_SIZE = 768 # For bert-base-uncased
RESNET_OUTPUT_SIZE = 2048 # For resnet50

# Fusion model settings
FUSION_HIDDEN_SIZE = 512
DROPOUT_RATE = 0.3

# --- Training Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 # Adjust based on your GPU memory / system capability
NUM_EPOCHS = 5 # Number of training epochs
LEARNING_RATE = 2e-5 # Learning rate for the optimizer (AdamW often works well)
# Consider separate learning rates if fine-tuning vs just training fusion layer

# Optional: Control freezing of pre-trained backbones
FREEZE_BERT = False # Set to False to fine-tune BERT
FREEZE_RESNET = False # Set to False to fine-tune ResNet

# --- Text Preprocessing ---
MAX_TOKEN_LEN = 128 # Max sequence length for BERT tokenizer

# --- Label Mapping ---
# Ensure keys exactly match the strings in your 'Sentiment' column
LABEL_MAP = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
# Create inverse map for reporting
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# --- Misc ---
RANDOM_SEED = 42 # For reproducibility in splits, etc.
NUM_WORKERS = 2 # Number of workers for DataLoader (adjust based on CPU cores)

print(f"Configuration loaded. Using device: {DEVICE}")
