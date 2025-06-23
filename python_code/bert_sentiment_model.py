import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

def create_modified_bert_model(num_classes=2, model_name="bert-base-uncased"):
    """
    Create a BERT model with frozen layers except the output layer,
    modified for binary sentiment classification.
    """
    # Load the pre-trained BERT model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    # Freeze all BERT layers except the classifier
    for name, param in model.named_parameters():
        if 'classifier' not in name:  # Freeze everything except the classifier
            param.requires_grad = False
        else:
            param.requires_grad = True  # Keep classifier trainable
    
    return model

def create_custom_bert_model(num_classes=2, model_name="bert-base-uncased"):
    """
    Alternative approach: Create a custom model with BERT as feature extractor
    """
    # Load BERT as feature extractor (without classification head)
    bert = AutoModel.from_pretrained(model_name)
    
    # Freeze all BERT parameters
    for param in bert.parameters():
        param.requires_grad = False
    
    # Create custom classifier
    class BertSentimentClassifier(nn.Module):
        def __init__(self, bert_model, num_classes=2, dropout=0.1):
            super(BertSentimentClassifier, self).__init__()
            self.bert = bert_model
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(768, num_classes)  # 768 is BERT's hidden size
            
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            # Get BERT outputs (frozen)
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # Use [CLS] token representation
            pooled_output = outputs.pooler_output
            
            # Apply dropout and classification
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            
            return logits
    
    return BertSentimentClassifier(bert, num_classes)

# Example usage
if __name__ == "__main__":
    # Method 1: Modify existing model
    print("Method 1: Modifying existing BERT model")
    model1 = create_modified_bert_model(num_classes=2)
    
    # Check which parameters are trainable
    trainable_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model1.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # Method 2: Custom model with BERT as feature extractor
    print("\nMethod 2: Custom model with BERT as feature extractor")
    model2 = create_custom_bert_model(num_classes=2)
    
    # Check which parameters are trainable
    trainable_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model2.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # Test with sample input
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    sample_text = "I love this movie!"
    inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
    
    # Test both models
    with torch.no_grad():
        outputs1 = model1(**inputs)
        outputs2 = model2(**inputs)
    
    print(f"\nSample output shape (Method 1): {outputs1.logits.shape}")
    print(f"Sample output shape (Method 2): {outputs2.shape}") 