#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Define paths
FEATURES_CACHE_PATH = 'resnet_features.npy'
DATA_CACHE_PATH = 'data_df.pkl'
INDICES_CACHE_PATH_SCALED = 'selected_indices_scaled.npy'
INDICES_CACHE_PATH_UNSCALED = 'selected_indices_unscaled.npy'
SCALER_CACHE_PATH = 'scaler.joblib'
MODEL_CACHE_PATH = 'knn_final_model.joblib'
EVALUATION_RESULTS_PATH = 'evaluation_results.joblib'

# Define emotions mapping
EMOTIONS = { 
    0: 'Angry',
    2: 'Happy',
    3: 'Neutral',
    4: 'Sad'
}

def generate_knn_roc_curves(X_test_selected, y_test, knn_model, name):
    """Generate ROC curves for k-NN models by manually calculating probability estimates."""
    print(f"Generating ROC curves for {name}...")
    
    try:
        # Get unique classes and prepare labels for binary classification
        unique_classes = np.sort(np.unique(y_test))
        n_classes = len(unique_classes)
        y_test_bin = label_binarize(y_test, classes=unique_classes)
        
        # Get distances and indices of nearest neighbors
        distances, indices = knn_model.kneighbors(X_test_selected)
        
        # Initialize probability estimates
        y_score = np.zeros((len(X_test_selected), n_classes))
        y_train = knn_model._y  # Get the training labels
        
        # Calculate probability estimates based on neighbor distances
        for i in range(len(X_test_selected)):
            neighbor_indices = indices[i]
            neighbor_distances = distances[i]
            
            # Avoid division by zero
            epsilon = 1e-10
            weights = 1.0 / (neighbor_distances + epsilon)
            
            # Count weighted votes for each class
            for j, class_val in enumerate(unique_classes):
                class_neighbors = np.where(y_train[neighbor_indices] == class_val)[0]
                if len(class_neighbors) > 0:
                    y_score[i, j] = weights[class_neighbors].sum()
            
            # Normalize to get probabilities
            row_sum = y_score[i].sum()
            if row_sum > 0:
                y_score[i] = y_score[i] / row_sum
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i, class_val in enumerate(unique_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        
        # Plot micro-average ROC curve
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)
        
        # Plot ROC curves for all classes
        colors = ['blue', 'red', 'green', 'orange']
        for i, (class_val, color) in enumerate(zip(unique_classes, colors)):
            emotion_name = EMOTIONS[class_val]
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{emotion_name} (AUC = {roc_auc[i]:.2f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save figure
        filename = f'roc_curves_{name.replace(" ", "_").lower()}.png'
        plt.savefig(filename)
        plt.close()
        
        print(f"Successfully saved ROC curves to {filename}")
        return True
        
    except Exception as e:
        import traceback
        print(f"Could not create ROC curves for {name}: {e}")
        traceback.print_exc()
        return False

def main():
    # Check if required files exist
    if not os.path.exists(EVALUATION_RESULTS_PATH):
        print(f"Evaluation results not found at {EVALUATION_RESULTS_PATH}")
        return
        
    # Load evaluation results
    evaluation_results = joblib.load(EVALUATION_RESULTS_PATH)
    
    # Load data and model for regenerating ROC curves
    print("Loading data and model...")
    features = np.load(FEATURES_CACHE_PATH)
    data_df = pd.read_pickle(DATA_CACHE_PATH)
    model = joblib.load(MODEL_CACHE_PATH)
    scaler = joblib.load(SCALER_CACHE_PATH)
    
    # Load selected indices
    selected_indices_scaled = np.load(INDICES_CACHE_PATH_SCALED)
    selected_indices_unscaled = np.load(INDICES_CACHE_PATH_UNSCALED)
    
    # Split data
    X = features
    y = data_df['Emotion'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features for scaled version
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select features
    X_train_selected_scaled = X_train_scaled[:, selected_indices_scaled]
    X_test_selected_scaled = X_test_scaled[:, selected_indices_scaled]
    
    X_train_selected_unscaled = X_train[:, selected_indices_unscaled]
    X_test_selected_unscaled = X_test[:, selected_indices_unscaled]
    
    # Generate ROC curves for scaled features
    print("Generating ROC curves for scaled features...")
    scaled_success = generate_knn_roc_curves(
        X_test_selected_scaled, 
        y_test, 
        model, 
        "Scaled Features"
    )
    
    # Create and train a new model for unscaled features
    print("Training model on unscaled features...")
    knn_unscaled = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn_unscaled.fit(X_train_selected_unscaled, y_train)
    
    # Generate ROC curves for unscaled features
    print("Generating ROC curves for unscaled features...")
    unscaled_success = generate_knn_roc_curves(
        X_test_selected_unscaled, 
        y_test, 
        knn_unscaled, 
        "Unscaled Features"
    )
    
    # Calculate metrics for unscaled features
    y_pred_unscaled = knn_unscaled.predict(X_test_selected_unscaled)
    from sklearn.metrics import accuracy_score, f1_score
    test_accuracy_unscaled = accuracy_score(y_test, y_pred_unscaled)
    test_f1_unscaled = f1_score(y_test, y_pred_unscaled, average='weighted')
    
    print(f"Unscaled features - Test Accuracy: {test_accuracy_unscaled:.4f}, F1: {test_f1_unscaled:.4f}")
    
    # Update evaluation results
    evaluation_results['test_accuracy_unscaled'] = test_accuracy_unscaled
    evaluation_results['test_f1_unscaled'] = test_f1_unscaled
    
    # Generate confusion matrix for unscaled features
    from sklearn.metrics import confusion_matrix
    unique_emotions = np.sort(np.unique(y))
    emotion_names = [EMOTIONS.get(e, f"Unknown-{e}") for e in unique_emotions]
    cm_unscaled = confusion_matrix(y_test, y_pred_unscaled, labels=unique_emotions)
    
    # Create confusion matrix figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_unscaled,
        annot=True,
        fmt='d',
        cmap='RdPu',
        xticklabels=emotion_names,
        yticklabels=emotion_names
    )
    plt.title('Confusion Matrix (Unscaled Features)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_unscaled.png')
    plt.close()
    
    # Update evaluation results
    evaluation_results['confusion_matrix_unscaled'] = cm_unscaled
    
    # Generate classification report
    from sklearn.metrics import classification_report
    classification_report_unscaled = classification_report(
        y_test,
        y_pred_unscaled,
        labels=unique_emotions,
        target_names=emotion_names,
        output_dict=True
    )
    evaluation_results['classification_report_unscaled'] = classification_report_unscaled
    
    # Save updated evaluation results
    joblib.dump(evaluation_results, EVALUATION_RESULTS_PATH)
    print(f"Updated evaluation results saved to {EVALUATION_RESULTS_PATH}")

if __name__ == "__main__":
    main()