# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import time
import os
from tqdm import tqdm

# Import components from local modules
from .config import (
    DEVICE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, DATA_FILE, IMAGE_DIR,
    LABEL_MAP, INV_LABEL_MAP, BERT_MODEL_NAME, NUM_CLASSES, RANDOM_SEED,
    NUM_WORKERS, FREEZE_BERT, FREEZE_RESNET, IMAGE_FILENAME_COL, TEXT_COL, SENTIMENT_COL
)
from .preprocess import MultimodalSentimentDataset, image_transform
from .models import MultimodalSentimentModel

# --- Seed for reproducibility ---
def set_seed(seed_value=RANDOM_SEED):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Training Function ---
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    """Trains the model for one epoch."""
    model.train() # Set model to training mode
    total_loss = 0
    correct_predictions = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False, total=len(data_loader))
    for batch in progress_bar:
        # Move batch data to the configured device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image=images
        )

        # Calculate loss
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        # Calculate accuracy
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        # Optional: Clip gradients to prevent exploding gradients
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
             scheduler.step() # Update learning rate scheduler

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / n_examples
    return avg_loss, accuracy

# --- Evaluation Function ---
def evaluate(model, data_loader, loss_fn, device, n_examples):
    """Evaluates the model on a given dataset."""
    model.eval() # Set model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculations during evaluation
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False, total=len(data_loader))
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=images
            )

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

            # Store predictions and labels for detailed report
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': loss.item()})


    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / n_examples
    report = classification_report(all_labels, all_preds, target_names=INV_LABEL_MAP.values(), zero_division=0)

    return avg_loss, accuracy, report

# --- Main Execution ---
def main():
    """Main function to orchestrate loading, training, and evaluation."""
    set_seed(RANDOM_SEED)

    print("--- Multimodal Sentiment Analysis ---")
    print(f"Using device: {DEVICE}")

    # --- 1. Check for Data and Load Metadata ---
    if not os.path.exists(DATA_FILE):
        print(f"\nError: Data file '{DATA_FILE}' not found in project root.")
        print("Please download the dataset from Kaggle and place it correctly (see README.md).")
        return
    if not os.path.exists(IMAGE_DIR):
        print(f"\nError: Image directory '{IMAGE_DIR}' not found in project root.")
        print("Please download the dataset and place it correctly (see README.md).")
        return

    print(f"\nLoading data mapping from: {DATA_FILE}")
    # Load the entire dataset metadata first for splitting
    try:
        full_df = pd.read_excel(DATA_FILE)
        # Basic validation
        required_cols = [IMAGE_FILENAME_COL, TEXT_COL, SENTIMENT_COL]
        if not all(col in full_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in full_df.columns]
            print(f"Error: Missing required columns in {DATA_FILE}: {missing}")
            return
        full_df['label'] = full_df[SENTIMENT_COL].map(LABEL_MAP)
        initial_len = len(full_df)
        full_df.dropna(subset=['label'], inplace=True)
        if len(full_df) < initial_len:
            print(f"Warning: Dropped {initial_len - len(full_df)} rows due to invalid sentiment labels.")
        full_df['label'] = full_df['label'].astype(int)

        print(f"Loaded {len(full_df)} valid entries.")

    except Exception as e:
        print(f"Error loading or processing {DATA_FILE}: {e}")
        return

    # --- 2. Split Data ---
    # Split into train+val and test sets (e.g., 80% train+val, 20% test)
    train_val_df, test_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=full_df['label'] # Ensure class distribution is similar in splits
    )
    # Split train+val into train and validation sets (e.g., 90% train, 10% val from train_val)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.1, # 10% of the 80% = 8% of total
        random_state=RANDOM_SEED,
        stratify=train_val_df['label']
    )
    print(f"Data split: Train ({len(train_df)}), Validation ({len(val_df)}), Test ({len(test_df)})")

    # --- 3. Initialize Tokenizer, Datasets, DataLoaders ---
    print(f"\nInitializing tokenizer: {BERT_MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    print("Creating datasets...")
    # We pass the dataframe subsets directly to the Dataset constructor
    # Note: The MultimodalSentimentDataset defined in preprocess.py handles path construction internally
    train_dataset = MultimodalSentimentDataset(
        dataframe=train_df, image_dir=IMAGE_DIR, tokenizer=tokenizer, transform=image_transform
    )
    val_dataset = MultimodalSentimentDataset(
        dataframe=val_df, image_dir=IMAGE_DIR, tokenizer=tokenizer, transform=image_transform
    )
    test_dataset = MultimodalSentimentDataset(
        dataframe=test_df, image_dir=IMAGE_DIR, tokenizer=tokenizer, transform=image_transform
    )

    # Check if datasets were created successfully (MultimodalSentimentDataset prints errors if loading fails)
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("Error: One or more datasets are empty. Cannot proceed with training. Check previous logs.")
        return

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- 4. Initialize Model, Optimizer, Loss, Scheduler ---
    print("\nInitializing model...")
    model = MultimodalSentimentModel(num_classes=NUM_CLASSES).to(DEVICE)

    print("Initializing optimizer and loss function...")
    # Define optimizer - optimize only parameters that require gradients
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, eps=1e-8)

    criterion = nn.CrossEntropyLoss().to(DEVICE) # Loss function

    # Optional: Learning rate scheduler
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, # Optional: set number of warmup steps
        num_training_steps=total_steps
    )

    # --- 5. Training Loop ---
    print("\n--- Starting Training ---")
    best_val_accuracy = 0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f'\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---')

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scheduler, len(train_dataset)
        )
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')

        # Validate
        val_loss, val_acc, val_report = evaluate(
            model, val_loader, criterion, DEVICE, len(val_dataset)
        )
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
        print("Validation Classification Report:\n", val_report)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch completed in {epoch_time:.2f} seconds.")

        # Save the best model based on validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            # Optional: Save model checkpoint
            # torch.save(model.state_dict(), 'best_multimodal_model.pth')
            # print("Saved best model checkpoint.")

    total_training_time = time.time() - start_time
    print(f"\n--- Training Finished ---")
    print(f"Total training time: {total_training_time:.2f} seconds.")
    print(f"Best validation accuracy achieved: {best_val_accuracy:.4f}")

    # --- 6. Final Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set ---")
    # Optional: Load best model if saved
    # model.load_state_dict(torch.load('best_multimodal_model.pth'))
    test_loss, test_acc, test_report = evaluate(
        model, test_loader, criterion, DEVICE, len(test_dataset)
    )

    print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    print("\nTest Set Classification Report:\n", test_report)
    print("---------------------------------")

# --- Run the main function ---
if __name__ == '__main__':
    main()
