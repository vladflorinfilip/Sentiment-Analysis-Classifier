# Replace the problematic plot_interactive_curves function with this working version
def plot_roc_curves(model, X_test, y_test, X_train=None, y_train=None):
    """
    Plot ROC curves and Precision-Recall curves for the model
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    # Get predictions
    y_test_pred_proba = model.predict(X_test).ravel()
    fpr_test, tpr_test, roc_thresholds_test = roc_curve(y_test, y_test_pred_proba)
    roc_auc_test = auc(fpr_test, tpr_test)

    if X_train is not None and y_train is not None:
        y_train_pred_proba = model.predict(X_train).ravel()
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
        roc_auc_train = auc(fpr_train, tpr_train)
    else:
        fpr_train, tpr_train, roc_auc_train = None, None, None

    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_test_pred_proba)
    pr_auc = auc(recall, precision)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot ROC curve
    ax1.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC (AUC = {roc_auc_test:.3f})')
    if fpr_train is not None:
        ax1.plot(fpr_train, tpr_train, color='blue', lw=2, linestyle='--', label=f'Train ROC (AUC = {roc_auc_train:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':')
    
    # Add threshold points
    thresholds_to_show = [0.2, 0.4, 0.5, 0.6, 0.8]
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    
    for i, threshold in enumerate(thresholds_to_show):
        idx = np.argmin(np.abs(roc_thresholds_test - threshold))
        ax1.scatter(fpr_test[idx], tpr_test[idx], color=colors[i], s=100, 
                   label=f'Threshold = {threshold:.1f}')

    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Plot Precision-Recall curve
    ax2.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # Add threshold points on PR curve
    for i, threshold in enumerate(thresholds_to_show):
        idx = np.argmin(np.abs(pr_thresholds - threshold))
        if idx < len(recall) and idx < len(precision):
            ax2.scatter(recall[idx], precision[idx], color=colors[i], s=100, 
                       label=f'Threshold = {threshold:.1f}')

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print metrics for different thresholds
    print("\nMetrics for different thresholds:")
    print("-" * 50)
    for threshold in thresholds_to_show:
        y_pred = (y_test_pred_proba >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        
        precision_at_threshold = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision_at_threshold * recall_at_threshold) / (precision_at_threshold + recall_at_threshold) if (precision_at_threshold + recall_at_threshold) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        print(f"Threshold = {threshold:.1f}:")
        print(f"  Precision: {precision_at_threshold:.3f}")
        print(f"  Recall: {recall_at_threshold:.3f}")
        print(f"  F1 Score: {f1_score:.3f}")
        print(f"  Accuracy: {accuracy:.3f}")
        print()

# Usage example:
# Replace the call to plot_interactive_curves with:
# plot_roc_curves(l2_model, X_test, y_test, X_train, y_train) 