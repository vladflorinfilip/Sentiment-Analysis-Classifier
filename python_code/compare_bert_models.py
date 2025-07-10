import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os
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

def load_frozen_bert_model(model_path):
    """Load the frozen BERT model"""
    print(f"Loading frozen BERT model from {model_path}")
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # Freeze all layers except classifier
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Frozen BERT - Trainable parameters: {trainable_params:,}")
    print(f"Frozen BERT - Total parameters: {total_params:,}")
    
    return model, tokenizer

def load_lora_model(base_model_path, lora_model_path):
    """Load the LoRA fine-tuned model"""
    print(f"Loading LoRA model from {lora_model_path}")
    
    # Load base model
    base_model = BertForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    tokenizer = BertTokenizer.from_pretrained(lora_model_path)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"LoRA BERT - Trainable parameters: {trainable_params:,}")
    print(f"LoRA BERT - Total parameters: {total_params:,}")
    
    return model, tokenizer

def evaluate_model(model, test_loader, device, model_name):
    """Evaluate a model and return metrics"""
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=['Negative', 'Positive'], output_dict=True)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': predictions,
        'true_labels': true_labels,
        'probabilities': probabilities
    }

def plot_confusion_matrices(frozen_results, lora_results, save_path=None):
    """Plot confusion matrices for both models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Frozen BERT confusion matrix
    cm_frozen = confusion_matrix(frozen_results['true_labels'], frozen_results['predictions'])
    sns.heatmap(cm_frozen, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'Frozen BERT\nAccuracy: {frozen_results["accuracy"]:.4f}')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # LoRA BERT confusion matrix
    cm_lora = confusion_matrix(lora_results['true_labels'], lora_results['predictions'])
    sns.heatmap(cm_lora, annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_title(f'LoRA BERT\nAccuracy: {lora_results["accuracy"]:.4f}')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to {save_path}")
    
    plt.show()

def plot_accuracy_comparison(frozen_results, lora_results, save_path=None):
    """Plot accuracy comparison"""
    models = ['Frozen BERT', 'LoRA BERT']
    accuracies = [frozen_results['accuracy'], lora_results['accuracy']]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy comparison saved to {save_path}")
    
    plt.show()

def analyze_predictions(frozen_results, lora_results):
    """Analyze where models agree and disagree"""
    frozen_preds = np.array(frozen_results['predictions'])
    lora_preds = np.array(lora_results['predictions'])
    true_labels = np.array(frozen_results['true_labels'])
    
    # Find where models agree and disagree
    agreement_mask = frozen_preds == lora_preds
    disagreement_mask = frozen_preds != lora_preds
    
    agreement_count = np.sum(agreement_mask)
    disagreement_count = np.sum(disagreement_mask)
    
    print(f"\nModel Agreement Analysis:")
    print(f"Agreement: {agreement_count} samples ({agreement_count/len(true_labels)*100:.2f}%)")
    print(f"Disagreement: {disagreement_count} samples ({disagreement_count/len(true_labels)*100:.2f}%)")
    
    # Analyze disagreement cases
    if disagreement_count > 0:
        disagreement_correct = np.sum((frozen_preds[disagreement_mask] == true_labels[disagreement_mask]) | 
                                     (lora_preds[disagreement_mask] == true_labels[disagreement_mask]))
        
        print(f"Disagreement cases where at least one model is correct: {disagreement_correct}")
        print(f"Disagreement cases where both models are wrong: {disagreement_count - disagreement_correct}")
    
    return {
        'agreement_count': agreement_count,
        'disagreement_count': disagreement_count,
        'agreement_percentage': agreement_count/len(true_labels)*100,
        'disagreement_percentage': disagreement_count/len(true_labels)*100
    }

def main():
    # Configuration
    DATA_PATH = "../train_data/reviews.csv"
    FROZEN_MODEL_PATH = "models/bert_sentiment_frozen"
    LORA_MODEL_PATH = "models/bert_lora_sentiment_20250210_120000"  # Update with your actual LoRA model path
    BASE_MODEL_PATH = "models/bert_sentiment_frozen"
    
    BATCH_SIZE = 16
    MAX_LENGTH = 256
    
    # Device detection
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    label_map = {'Positive': 1, 'Negative': 0}
    df['label'] = df['sentiment'].map(label_map)
    
    # Use a test set for evaluation
    _, test_texts, _, test_labels = train_test_split(
        df['review'].values, 
        df['label'].values, 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    print(f"Test samples: {len(test_texts)}")
    print(f"Class distribution - Test: {np.bincount(test_labels)}")
    
    # Load models
    frozen_model, frozen_tokenizer = load_frozen_bert_model(FROZEN_MODEL_PATH)
    lora_model, lora_tokenizer = load_lora_model(BASE_MODEL_PATH, LORA_MODEL_PATH)
    
    # Create test datasets
    frozen_dataset = SentimentDataset(test_texts, test_labels, frozen_tokenizer, MAX_LENGTH)
    lora_dataset = SentimentDataset(test_texts, test_labels, lora_tokenizer, MAX_LENGTH)
    
    # Create test loaders
    frozen_loader = DataLoader(frozen_dataset, batch_size=BATCH_SIZE, shuffle=False)
    lora_loader = DataLoader(lora_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Move models to device
    frozen_model.to(DEVICE)
    lora_model.to(DEVICE)
    
    # Evaluate models
    print("\nEvaluating models...")
    frozen_results = evaluate_model(frozen_model, frozen_loader, DEVICE, "Frozen BERT")
    lora_results = evaluate_model(lora_model, lora_loader, DEVICE, "LoRA BERT")
    
    # Print results
    print(f"\n=== RESULTS ===")
    print(f"Frozen BERT Accuracy: {frozen_results['accuracy']:.4f}")
    print(f"LoRA BERT Accuracy: {lora_results['accuracy']:.4f}")
    print(f"Improvement: {lora_results['accuracy'] - frozen_results['accuracy']:.4f}")
    
    print(f"\n=== FROZEN BERT DETAILED RESULTS ===")
    print(f"Precision: {frozen_results['classification_report']['weighted avg']['precision']:.4f}")
    print(f"Recall: {frozen_results['classification_report']['weighted avg']['recall']:.4f}")
    print(f"F1-Score: {frozen_results['classification_report']['weighted avg']['f1-score']:.4f}")
    
    print(f"\n=== LORA BERT DETAILED RESULTS ===")
    print(f"Precision: {lora_results['classification_report']['weighted avg']['precision']:.4f}")
    print(f"Recall: {lora_results['classification_report']['weighted avg']['recall']:.4f}")
    print(f"F1-Score: {lora_results['classification_report']['weighted avg']['f1-score']:.4f}")
    
    # Analyze predictions
    agreement_analysis = analyze_predictions(frozen_results, lora_results)
    
    # Create visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot confusion matrices
    cm_save_path = f"../data/model_comparison_confusion_matrices_{timestamp}.png"
    plot_confusion_matrices(frozen_results, lora_results, cm_save_path)
    
    # Plot accuracy comparison
    acc_save_path = f"../data/model_comparison_accuracy_{timestamp}.png"
    plot_accuracy_comparison(frozen_results, lora_results, acc_save_path)
    
    # Save comprehensive results
    results = {
        "timestamp": timestamp,
        "frozen_bert": {
            "accuracy": frozen_results['accuracy'],
            "classification_report": frozen_results['classification_report']
        },
        "lora_bert": {
            "accuracy": lora_results['accuracy'],
            "classification_report": lora_results['classification_report']
        },
        "improvement": lora_results['accuracy'] - frozen_results['accuracy'],
        "agreement_analysis": agreement_analysis
    }
    
    results_path = f"../data/model_comparison_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print("Model comparison completed!")

if __name__ == "__main__":
    main() 