import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class SentimentDataset(Dataset):
    """Custom Dataset for sentiment analysis"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_prepare_data(data_path, test_size=0.2, random_state=42):
    """Load and prepare the dataset"""
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Convert sentiment labels to numeric
    label_map = {'Positive': 1, 'Negative': 0}
    df['label'] = df['sentiment'].map(label_map)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['review'].values, 
        df['label'].values, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']
    )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Class distribution - Train: {np.bincount(train_labels)}")
    print(f"Class distribution - Val: {np.bincount(val_labels)}")
    
    return train_texts, val_texts, train_labels, val_labels

def create_model_and_tokenizer(model_name="bert-base-uncased", num_classes=2):
    """Create model with frozen layers except the last one"""
    print(f"Loading {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    # Freeze all layers except the classifier
    print("Freezing BERT layers...")
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer

def train_model(model, train_loader, val_loader, tokenizer, 
                num_epochs=3, learning_rate=2e-5, device='cuda'):
    """Train the model"""
    print(f"Moving model to device: {device}")
    model.to(device)
    
    # Optimizer (only for trainable parameters)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_progress = tqdm(train_loader, desc="Training")
        
        for batch in train_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="Validation")
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                
                # Move predictions and labels to CPU for numpy conversion
                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
                
                val_progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate accuracy
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        val_accuracies.append(val_accuracy)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
    
    return train_losses, val_losses, val_accuracies

def plot_training_history(train_losses, val_losses, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(val_accuracies, label='Val Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('data/bert_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_model(model, tokenizer, save_path):
    """Save the trained model"""
    print(f"Saving model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model saved successfully!")

def main():
    # Configuration
    DATA_PATH = "train_data/reviews.csv"
    MODEL_NAME = "bert-base-uncased"
    BATCH_SIZE = 16
    MAX_LENGTH = 256
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    # Device detection for macOS (MPS) and other platforms
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'
    
    print(f"Using device: {DEVICE}")
    if DEVICE == 'mps':
        print("Using Apple Metal Performance Shaders (MPS) for GPU acceleration")
    elif DEVICE == 'cuda':
        print("Using CUDA for GPU acceleration")
    else:
        print("Using CPU for training")
    
    # Load and prepare data
    train_texts, val_texts, train_labels, val_labels = load_and_prepare_data(DATA_PATH)
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(MODEL_NAME)
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train the model
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, tokenizer, 
        num_epochs=EPOCHS, learning_rate=LEARNING_RATE, device=DEVICE
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_accuracies)
    
    # Save the model
    save_path = "models/bert_sentiment_frozen"
    os.makedirs("models", exist_ok=True)
    save_model(model, tokenizer, save_path)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 