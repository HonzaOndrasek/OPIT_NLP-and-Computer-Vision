# src/models.py

import torch
import torch.nn as nn
from transformers import BertModel, AutoConfig
from torchvision.models import resnet50, ResNet50_Weights

# Import configuration variables
from .config import (
    BERT_MODEL_NAME,
    NUM_CLASSES,
    DROPOUT_RATE,
    FREEZE_BERT,
    FREEZE_RESNET,
    FUSION_HIDDEN_SIZE,
    BERT_OUTPUT_SIZE, # Expected size from config for verification
    RESNET_OUTPUT_SIZE # Expected size from config for verification
)


class MultimodalSentimentModel(nn.Module):
    """
    Multimodal model combining features from BERT (text) and ResNet (image).
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE,
                 bert_model_name=BERT_MODEL_NAME,
                 freeze_bert=FREEZE_BERT, freeze_resnet=FREEZE_RESNET,
                 fusion_hidden_size=FUSION_HIDDEN_SIZE):
        """
        Args:
            num_classes (int): Number of output sentiment classes.
            dropout_rate (float): Dropout probability for the classification head.
            bert_model_name (str): Name of the pre-trained BERT model to load.
            freeze_bert (bool): Whether to freeze the weights of the BERT model.
            freeze_resnet (bool): Whether to freeze the weights of the ResNet model.
            fusion_hidden_size (int): Size of the hidden layer in the fusion classifier.
        """
        super().__init__()
        self.num_classes = num_classes
        self.freeze_bert = freeze_bert
        self.freeze_resnet = freeze_resnet

        # --- Load Pre-trained BERT ---
        print(f"Loading BERT model: {bert_model_name}")
        self.bert_config = AutoConfig.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        # Verify BERT output size matches config (optional but good practice)
        if self.bert_config.hidden_size != BERT_OUTPUT_SIZE:
             print(f"Warning: BERT hidden size ({self.bert_config.hidden_size}) does not match config ({BERT_OUTPUT_SIZE}). Using actual size.")
             self.bert_feature_size = self.bert_config.hidden_size
        else:
             self.bert_feature_size = BERT_OUTPUT_SIZE

        # Freeze BERT layers if specified
        if self.freeze_bert:
            print("Freezing BERT model weights.")
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
             print("BERT model weights will be fine-tuned.")


        # --- Load Pre-trained ResNet ---
        print("Loading ResNet50 model (pre-trained on ImageNet).")
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Get the input feature size of the original ResNet classifier
        resnet_fc_in_features = self.resnet.fc.in_features
        # Verify ResNet output size matches config (optional but good practice)
        if resnet_fc_in_features != RESNET_OUTPUT_SIZE:
             print(f"Warning: ResNet feature size ({resnet_fc_in_features}) does not match config ({RESNET_OUTPUT_SIZE}). Using actual size.")
             self.resnet_feature_size = resnet_fc_in_features
        else:
             self.resnet_feature_size = RESNET_OUTPUT_SIZE
        # Remove the final classification layer (fc) by replacing it with Identity
        self.resnet.fc = nn.Identity()

        # Freeze ResNet layers if specified
        if self.freeze_resnet:
            print("Freezing ResNet model weights.")
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            print("ResNet model weights will be fine-tuned.")

        # --- Fusion and Classification Head ---
        print("Initializing fusion and classification head.")
        self.fusion_layer_norm = nn.LayerNorm(self.bert_feature_size + self.resnet_feature_size) # Optional LayerNorm
        self.fusion = nn.Sequential(
            # Optional: Layer Normalization before MLP
            # nn.LayerNorm(self.bert_feature_size + self.resnet_feature_size),
            nn.Linear(self.bert_feature_size + self.resnet_feature_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_size, num_classes)
        )
        print(f"Model initialized. Combined feature size: {self.bert_feature_size + self.resnet_feature_size}, Fusion hidden size: {fusion_hidden_size}, Output classes: {num_classes}")

    def forward(self, input_ids, attention_mask, image):
        """
        Forward pass through the multimodal model.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs for text (batch_size, seq_len).
            attention_mask (torch.Tensor): Tensor of attention masks for text (batch_size, seq_len).
            image (torch.Tensor): Tensor of preprocessed images (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Raw logits output from the classifier (batch_size, num_classes).
        """
        # --- Text Feature Extraction ---
        # If BERT is frozen, disable gradient calculation for this part
        if self.freeze_bert:
            with torch.no_grad():
                bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooler output which represents the entire sequence
        bert_features = bert_outputs.pooler_output # Shape: (batch_size, bert_hidden_size)

        # --- Image Feature Extraction ---
        # If ResNet is frozen, disable gradient calculation
        if self.freeze_resnet:
            with torch.no_grad():
                # Set ResNet to eval mode if frozen to disable dropout/batchnorm updates, even if model.train() was called
                self.resnet.eval()
                image_features = self.resnet(image) # Shape: (batch_size, resnet_feature_size)
        else:
            # Ensure ResNet is in the correct mode (train/eval) based on the parent model's state
            # No need to manually set self.resnet.train() or self.resnet.eval() here if fine-tuning,
            # as it inherits the mode from the parent MultimodalSentimentModel.
            image_features = self.resnet(image) # Shape: (batch_size, resnet_feature_size)


        # --- Feature Fusion ---
        # Concatenate text and image features along the feature dimension (dim=1)
        combined_features = torch.cat((bert_features, image_features), dim=1)

        # Optional: Apply Layer Normalization to the combined features
        # combined_features = self.fusion_layer_norm(combined_features)

        # --- Classification ---
        # Pass the combined features through the final classification layers
        logits = self.fusion(combined_features) # Shape: (batch_size, num_classes)

        return logits


# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    print("Testing MultimodalSentimentModel initialization...")

    # Create a dummy model instance
    try:
        model = MultimodalSentimentModel()
        print("\nModel instance created successfully.")

        # Print model structure (optional)
        # print("\nModel Architecture:")
        # print(model)

        # Test forward pass with dummy data
        print("\nTesting forward pass with dummy data...")
        batch_size = 4
        seq_len = config.MAX_TOKEN_LEN
        img_size = config.IMG_SIZE
        dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        dummy_attention_mask = torch.ones((batch_size, seq_len))
        dummy_images = torch.randn(batch_size, 3, img_size, img_size)

        # Put model in evaluation mode for the test forward pass
        model.eval()
        with torch.no_grad():
            dummy_output = model(dummy_input_ids, dummy_attention_mask, dummy_images)

        print("Dummy Output Shape:", dummy_output.shape)
        print("Expected Output Shape:", (batch_size, config.NUM_CLASSES))
        if dummy_output.shape == (batch_size, config.NUM_CLASSES):
             print("Forward pass test successful!")
        else:
             print("Forward pass test FAILED: Output shape mismatch.")


    except Exception as e:
        print(f"\nError during model testing: {e}")
        import traceback
        traceback.print_exc()
