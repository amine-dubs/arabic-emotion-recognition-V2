import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import time
import gc
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
import datetime

# Create a results directory with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("classifier_comparison_results", timestamp)
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved in: {results_dir}")

# Define paths for data files
DATA_CACHE_PATH = 'data_df.pkl'
FEATURES_CACHE_PATH = 'resnet_features.npy'
SCALER_CACHE_PATH = 'scaler.joblib'
INDICES_CACHE_PATH_SCALED = 'selected_indices_scaled.npy'

# Load the data and selected features
print("Loading data and features...")
if os.path.exists(FEATURES_CACHE_PATH) and os.path.exists(DATA_CACHE_PATH):
    resnet_features = np.load(FEATURES_CACHE_PATH)
    Data = pd.read_pickle(DATA_CACHE_PATH)
    print(f"Loaded {resnet_features.shape[0]} samples with {resnet_features.shape[1]} features each")
else:
    print("Error: Cache not found. Run the main script first.")
    exit()

# Load the scaler
if os.path.exists(SCALER_CACHE_PATH):
    scaler = joblib.load(SCALER_CACHE_PATH)
    print("Loaded scaler successfully")
else:
    print("Error: Scaler not found. Run the main script first.")
    exit()

# Load the selected indices
if os.path.exists(INDICES_CACHE_PATH_SCALED):
    selected_indices = np.load(INDICES_CACHE_PATH_SCALED)
    print(f"Loaded {len(selected_indices)} selected feature indices")
else:
    print("Error: Selected indices not found. Run the main script first.")
    exit()

# Define the emotion mapping
EMOTIONS = { 
    0: 'Angry',
    2: 'Happy',
    3: 'Neutral',
    4: 'Sad'
}

# Extract features and labels
X = resnet_features
y = Data['Emotion'].values

# Split into training and testing sets (keeping the same split as the original)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply feature selection
X_train_selected = X_train_scaled[:, selected_indices]
X_test_selected = X_test_scaled[:, selected_indices]

# Define classifiers to evaluate
classifiers = {
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan', algorithm='auto', n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
    'SVM': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
}

# Let's limit feature count for experiment
feature_counts = [100, 200, 500, len(selected_indices)]  # Original count and reduced counts

results = []

# Create subdirectories for confusion matrices and ROC curves
cm_dir = os.path.join(results_dir, "confusion_matrices")
roc_dir = os.path.join(results_dir, "roc_curves")
os.makedirs(cm_dir, exist_ok=True)
os.makedirs(roc_dir, exist_ok=True)

for feature_count in feature_counts:
    # Select top N features only
    feature_subset = selected_indices[:feature_count]
    X_train_subset = X_train_scaled[:, feature_subset]
    X_test_subset = X_test_scaled[:, feature_subset]
    
    print(f"\nEvaluating with top {feature_count} features:")
    
    for name, classifier in classifiers.items():
        start_time = time.time()
        print(f"\nTraining {name}...")
        classifier.fit(X_train_subset, y_train)
        train_time = time.time() - start_time
        
        # Training accuracy
        train_pred = classifier.predict(X_train_subset)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        # Test accuracy
        start_time = time.time()
        test_pred = classifier.predict(X_test_subset)
        test_time = time.time() - start_time
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, test_pred)
        f1 = f1_score(y_test, test_pred, average='weighted')
        precision = precision_score(y_test, test_pred, average='weighted')
        recall = recall_score(y_test, test_pred, average='weighted')
        
        print(f"{name} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"{name} - F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"{name} - Training Time: {train_time:.2f}s, Prediction Time: {test_time:.2f}s")
        
        # Store results
        results.append({
            'Classifier': name,
            'Features': feature_count,
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall,
            'Training Time': train_time,
            'Prediction Time': test_time
        })
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, test_pred)
        plt.figure(figsize=(8, 6))
        emotion_names = [EMOTIONS[e] for e in sorted(np.unique(y))]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=emotion_names, 
                   yticklabels=emotion_names)
        plt.title(f'Confusion Matrix - {name} with {feature_count} features')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        cm_filename = f'confusion_matrix_{name.replace(" ", "_")}_f{feature_count}.png'
        plt.savefig(os.path.join(cm_dir, cm_filename))
        plt.close()
        
        # Generate ROC curves (with error handling)
        try:
            # Check if the classifier supports probability estimates
            if hasattr(classifier, "predict_proba"):
                plt.figure(figsize=(10, 8))
                
                # Get unique classes and prepare binarized labels
                unique_classes = sorted(np.unique(y))
                n_classes = len(unique_classes)
                
                # Create mapping from actual class codes to indices
                class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
                
                # Binarize the test labels
                y_test_bin = np.zeros((len(y_test), n_classes))
                for i, cls in enumerate(y_test):
                    y_test_bin[i, class_mapping[cls]] = 1
                
                # Get probability predictions
                y_score = classifier.predict_proba(X_test_subset)
                
                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i, cls in enumerate(unique_classes):
                    cls_idx = class_mapping[cls]
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, cls_idx], y_score[:, cls_idx])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Plot ROC curves
                colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
                for i, cls in enumerate(unique_classes):
                    emotion_name = EMOTIONS[cls]
                    plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                             label=f'{emotion_name} (AUC = {roc_auc[i]:.2f})')
                
                # Plot diagonal line
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curves - {name} with {feature_count} features')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                roc_filename = f'roc_curves_{name.replace(" ", "_")}_f{feature_count}.png'
                plt.savefig(os.path.join(roc_dir, roc_filename))
                plt.close()
                print(f"ROC curves generated successfully for {name}")
            else:
                print(f"Skipping ROC curve for {name} - No probability prediction support")
        except Exception as e:
            print(f"Could not generate ROC curves for {name}: {str(e)}")

# Create a summary dataframe
results_df = pd.DataFrame(results)

# Save results to CSV
results_csv = os.path.join(results_dir, 'classifier_comparison_results.csv')
results_df.to_csv(results_csv, index=False)
print(f"\nSaved results to {results_csv}")

# Create summary plots
plt.figure(figsize=(12, 8))
sns.barplot(data=results_df, x='Classifier', y='Test Accuracy', hue='Features')
plt.title('Test Accuracy by Classifier and Feature Count')
plt.xticks(rotation=45)
plt.tight_layout()
test_acc_plot = os.path.join(results_dir, 'test_accuracy_comparison.png')
plt.savefig(test_acc_plot)
plt.close()

plt.figure(figsize=(12, 8))
sns.barplot(data=results_df, x='Classifier', y='F1 Score', hue='Features')
plt.title('F1 Score by Classifier and Feature Count')
plt.xticks(rotation=45)
plt.tight_layout()
f1_plot = os.path.join(results_dir, 'f1_score_comparison.png')
plt.savefig(f1_plot)
plt.close()

# Create additional plots for time comparison
plt.figure(figsize=(12, 8))
sns.barplot(data=results_df, x='Classifier', y='Training Time', hue='Features')
plt.title('Training Time by Classifier and Feature Count')
plt.xticks(rotation=45)
plt.ylabel('Time (seconds)')
plt.tight_layout()
train_time_plot = os.path.join(results_dir, 'training_time_comparison.png')
plt.savefig(train_time_plot)
plt.close()

plt.figure(figsize=(12, 8))
sns.barplot(data=results_df, x='Classifier', y='Prediction Time', hue='Features')
plt.title('Prediction Time by Classifier and Feature Count')
plt.xticks(rotation=45)
plt.ylabel('Time (seconds)')
plt.tight_layout()
pred_time_plot = os.path.join(results_dir, 'prediction_time_comparison.png')
plt.savefig(pred_time_plot)
plt.close()

# Create a summary HTML report
html_report = os.path.join(results_dir, 'comparison_report.html')
with open(html_report, 'w') as f:
    f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Classifier Comparison Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #2c3e50; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        img { max-width: 100%; height: auto; margin: 10px 0; }
        .section { margin-bottom: 30px; }
        .image-gallery { display: flex; flex-wrap: wrap; gap: 10px; }
        .image-card { border: 1px solid #ddd; padding: 10px; border-radius: 5px; width: 300px; }
    </style>
</head>
<body>
    <h1>Arabic Emotion Recognition - Classifier Comparison</h1>
    <p>Generated on: ''' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '''</p>
    
    <div class="section">
        <h2>Performance Summary</h2>
        <p>Top performing classifiers by test accuracy:</p>
    ''')
    
    # Add top performers
    top_performers = results_df.sort_values(by='Test Accuracy', ascending=False).head(5)
    f.write("<table>")
    f.write("<tr><th>Rank</th><th>Classifier</th><th>Features</th><th>Test Accuracy</th><th>F1 Score</th><th>Training Time (s)</th></tr>")
    for i, (_, row) in enumerate(top_performers.iterrows()):
        f.write(f"<tr><td>{i+1}</td><td>{row['Classifier']}</td><td>{row['Features']}</td>")
        f.write(f"<td>{row['Test Accuracy']:.4f}</td><td>{row['F1 Score']:.4f}</td><td>{row['Training Time']:.2f}</td></tr>")
    f.write("</table>")
    
    # Add summary plots
    f.write('''
    <div class="section">
        <h2>Summary Plots</h2>
        <img src="test_accuracy_comparison.png" alt="Test Accuracy Comparison">
        <img src="f1_score_comparison.png" alt="F1 Score Comparison">
        <img src="training_time_comparison.png" alt="Training Time Comparison">
        <img src="prediction_time_comparison.png" alt="Prediction Time Comparison">
    </div>
    
    <div class="section">
        <h2>Confusion Matrices</h2>
        <p>Click on images to view larger versions</p>
        <div class="image-gallery">
    ''')
    
    # Add confusion matrix images
    for filename in sorted(os.listdir(cm_dir)):
        if filename.endswith('.png'):
            relative_path = os.path.join('confusion_matrices', filename)
            classifier_name = filename.replace('confusion_matrix_', '').replace('.png', '').replace('_', ' ')
            f.write(f'<div class="image-card"><a href="{relative_path}" target="_blank"><img src="{relative_path}" alt="{classifier_name}"></a>')
            f.write(f'<p>{classifier_name}</p></div>')
    
    f.write('''
        </div>
    </div>
    
    <div class="section">
        <h2>ROC Curves</h2>
        <p>Click on images to view larger versions</p>
        <div class="image-gallery">
    ''')
    
    # Add ROC curve images
    for filename in sorted(os.listdir(roc_dir)):
        if filename.endswith('.png'):
            relative_path = os.path.join('roc_curves', filename)
            classifier_name = filename.replace('roc_curves_', '').replace('.png', '').replace('_', ' ')
            f.write(f'<div class="image-card"><a href="{relative_path}" target="_blank"><img src="{relative_path}" alt="{classifier_name}"></a>')
            f.write(f'<p>{classifier_name}</p></div>')
    
    f.write('''
        </div>
    </div>
    
    <div class="section">
        <h2>Complete Results</h2>
        <p>See <a href="classifier_comparison_results.csv">classifier_comparison_results.csv</a> for the complete results data.</p>
    </div>
</body>
</html>
    ''')

print(f"\nResults saved to {results_dir}")
print(f"Summary report generated at: {html_report}")
print("Open the HTML report to view a comprehensive summary of the comparison results.")