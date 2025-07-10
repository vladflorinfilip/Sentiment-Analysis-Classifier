import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

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

def create_lora_model(base_model_path, num_labels=2):
    """Create a BERT model with LoRA adapters"""
    print("Loading base BERT model...")
    
    # Load the base model and tokenizer
    model = BertForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    tokenizer = BertTokenizer.from_pretrained(base_model_path)
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,  # Rank of the LoRA matrices
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,  # Dropout probability for LoRA layers
        target_modules=["query", "value"],  # Which attention modules to apply LoRA to
        bias="none",  # Whether to train bias terms
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model_with_lora(model, train_loader, val_loader, tokenizer, num_epochs=3, learning_rate=2e-4, device='cuda'):
    """Train the model with LoRA adapters"""
    model.to(device)

    # Optimizer (only for trainable parameters)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []

    print(f"Starting LoRA fine-tuning for {num_epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

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
            
            # Gradient clipping
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

def plot_training_history(train_losses, val_losses, val_accuracies, save_path=None):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()

def save_lora_model(model, tokenizer, save_path):
    """Save the LoRA model"""
    print(f"Saving LoRA model to {save_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save the LoRA adapters
    model.save_pretrained(save_path)
    
    # Save the tokenizer
    tokenizer.save_pretrained(save_path)
    
    # Save training configuration
    config = {
        "model_type": "bert_lora_sentiment",
        "base_model": "bert-base-uncased",
        "num_labels": 2,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["query", "value"],
            "bias": "none"
        },
        "training_config": {
            "learning_rate": 2e-4,
            "epochs": 3,
            "batch_size": 16,
            "max_length": 256
        }
    }
    
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("LoRA model saved successfully!")

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=['Negative', 'Positive'])
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return accuracy, report

def main():
    # Configuration
    DATA_PATH = "../train_data/reviews.csv"
    BASE_MODEL_PATH = "models/bert_sentiment_frozen"  # Your existing frozen BERT model
    BATCH_SIZE = 16
    MAX_LENGTH = 256
    EPOCHS = 3
    LEARNING_RATE = 2e-4  # Higher learning rate for LoRA

    # Device detection
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Load and prepare data
    train_texts, val_texts, train_labels, val_labels = load_and_prepare_data(DATA_PATH)
    
    # Create LoRA model
    model, tokenizer = create_lora_model(BASE_MODEL_PATH, num_labels=2)
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train the model with LoRA
    train_losses, val_losses, val_accuracies = train_model_with_lora(
        model, train_loader, val_loader, tokenizer, 
        num_epochs=EPOCHS, learning_rate=LEARNING_RATE, device=DEVICE
    )
    
    # Plot training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_save_path = f"../data/bert_lora_training_history_{timestamp}.png"
    plot_training_history(train_losses, val_losses, val_accuracies, plot_save_path)
    
    # Save the LoRA model
    save_path = f"models/bert_lora_sentiment_{timestamp}"
    save_lora_model(model, tokenizer, save_path)
    
    # Evaluate on validation set
    print("\nFinal evaluation on validation set:")
    accuracy, report = evaluate_model(model, val_loader, DEVICE)
    
    # Save evaluation results
    results = {
        "model_type": "bert_lora_sentiment",
        "timestamp": timestamp,
        "final_accuracy": accuracy,
        "training_history": {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        },
        "classification_report": report
    }
    
    results_path = f"../data/bert_lora_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print("LoRA fine-tuning completed!")

if __name__ == "__main__":
    main() 