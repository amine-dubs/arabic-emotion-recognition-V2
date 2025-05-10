#!/usr/bin/env python
# coding: utf-8

# This script uses EO (Equilibrium Optimizer) with Random Forest instead of KNN
# It includes validation split in addition to train/test

import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import time
import joblib
import gc
import argparse
import sys

print("Script started")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

# Imports for the model and algorithms
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize

# Import EO instead of SCA
from mealpy import EO
from mealpy.utils.space import BinaryVar
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier  # Random Forest instead of KNN
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Import ROC curve related metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# --- Cache File Paths ---
FEATURES_CACHE_PATH = 'resnet_features.npy'
DATA_CACHE_PATH = 'data_df.pkl'
INDICES_CACHE_PATH_SCALED = 'selected_indices_scaled_eo_rf.npy'
INDICES_CACHE_PATH_UNSCALED = 'selected_indices_unscaled_eo_rf.npy'
SCALER_CACHE_PATH = 'scaler_rf.joblib'
MODEL_CACHE_PATH = 'rf_final_model.joblib'
EVALUATION_RESULTS_PATH = 'evaluation_results_rf.joblib'
EO_HISTORY_SCALED_PATH = 'eo_rf_history_scaled.npy'
EO_HISTORY_UNSCALED_PATH = 'eo_rf_history_unscaled.npy'
INDICES_CACHE_PATH = INDICES_CACHE_PATH_SCALED  # Default to scaled indices
# ------------------------

EMOTIONS = {0: 'Angry',
           2: 'Happy',
           3: 'Neutral',
           4: 'Sad'}

DATA_PATH = './Data'
SAMPLE_RATE = 16000
DURATION = 3  # seconds
RESNET_INPUT_SHAPE = (224, 224, 3)

# --- Check for cached features and DataFrame ---
if os.path.exists(FEATURES_CACHE_PATH) and os.path.exists(DATA_CACHE_PATH):
    print(f"Loading cached features from {FEATURES_CACHE_PATH}...")
    resnet_features = np.load(FEATURES_CACHE_PATH)
    print(f"Loading cached DataFrame from {DATA_CACHE_PATH}...")
    Data = pd.read_pickle(DATA_CACHE_PATH)
    print("Loaded features shape:", resnet_features.shape)
    print("Loaded DataFrame head:\n", Data.head())
    # Skip feature extraction
    SKIP_FEATURE_EXTRACTION = True
else:
    print("Cache not found. Please run the feature extraction script first.")
    exit()
# ---------------------------------------------

# --- Check for cached model ---
if os.path.exists(MODEL_CACHE_PATH) and os.path.exists(SCALER_CACHE_PATH):
    print(f"Loading cached model from {MODEL_CACHE_PATH}...")
    rf_final = joblib.load(MODEL_CACHE_PATH)
    scaler = joblib.load(SCALER_CACHE_PATH)
    # Skip training
    SKIP_TRAINING = True
else:
    print("No cached model found. Will train a new model.")
    SKIP_TRAINING = False
# ---------------------------------------------

# # 4. Data Splitting and Scaling

# Ensure resnet_features is not empty before proceeding
if resnet_features.shape[0] == 0 or resnet_features.shape[0] != len(Data):
     print("Error: Feature array is empty or does not match DataFrame length. Exiting.")
     exit()

# --- Added: Shuffle features and corresponding labels ---
print("Shuffling features and labels before splitting...")
# Ensure reproducibility of the shuffle
permutation_indices = np.random.RandomState(seed=42).permutation(len(Data))
resnet_features = resnet_features[permutation_indices]
shuffled_emotions = Data['Emotion'].iloc[permutation_indices]
# --- End of shuffle ---

X = resnet_features
y = shuffled_emotions.values  # Use the shuffled emotions

# Split data into training, validation, and testing sets (70% train, 15% validation, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Original data shape:", X.shape, y.shape)
print("Training data shape:", X_train.shape, y_train.shape)
print("Validation data shape:", X_val.shape, y_val.shape)
print("Test data shape:", X_test.shape, y_test.shape)

# Store unscaled data for later use
X_train_unscaled = X_train.copy()
X_val_unscaled = X_val.copy()
X_test_unscaled = X_test.copy()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Free up memory
del X, X_temp, y_temp, resnet_features
gc.collect()

# # 5. Feature Selection using Sine Cosine Algorithm (SCA) with Random Forest

# Check if cached indices exist
if (os.path.exists(INDICES_CACHE_PATH_SCALED) and 
    os.path.exists(INDICES_CACHE_PATH_UNSCALED) and
    os.path.exists(EO_HISTORY_SCALED_PATH) and 
    os.path.exists(EO_HISTORY_UNSCALED_PATH)):
    print(f"Loading cached selected indices (EO with Random Forest)...")
    selected_feature_indices_scaled = np.load(INDICES_CACHE_PATH_SCALED)
    selected_feature_indices_unscaled = np.load(INDICES_CACHE_PATH_UNSCALED)
    eo_history_scaled = np.load(EO_HISTORY_SCALED_PATH)
    eo_history_unscaled = np.load(EO_HISTORY_UNSCALED_PATH)
    
    # Use scaled indices for backward compatibility
    selected_feature_indices = selected_feature_indices_scaled
    
    num_features_total = X_train_scaled.shape[1]
    print(f"Loaded {len(selected_feature_indices_scaled)} scaled indices and {len(selected_feature_indices_unscaled)} unscaled indices (EO-RF).")
    SKIP_EO = True
else:
    print("Cache not found. Running EO with Random Forest on both scaled and unscaled data...")
    SKIP_EO = False

if not SKIP_EO:
    # --- Common EO parameters ---
    num_features_total = X_train_scaled.shape[1]
    epoch = 50
    pop_size = 20
    
    # Create a list of BinaryVar objects, one for each feature
    binary_bounds = [BinaryVar() for _ in range(num_features_total)]
    
    # --- Define fitness function for SCALED data ---    
    # Random Forest instead of KNN
    rf_eval_scaled = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    )
    
    def fitness_function_scaled(solution):
        selected_indices = np.where(solution == 1)[0]
        num_selected = len(selected_indices)
        
        if num_selected == 0:
            return 1.0  # Penalize solutions with no features
        
        # Select features from the datasets
        X_train_subset = X_train_scaled[:, selected_indices]
        X_val_subset = X_val_scaled[:, selected_indices]
        
        # Train Random Forest on the TRAINING subset
        rf_eval_scaled.fit(X_train_subset, y_train)
        
        # Evaluate on VALIDATION set
        y_pred_val = rf_eval_scaled.predict(X_val_subset)
        accuracy = accuracy_score(y_val, y_pred_val)
        
        # Calculate fitness value (minimize this)
        # Balance between accuracy and number of features
        fitness = 0.99 * (1 - accuracy) + 0.01 * (num_selected / num_features_total)
        return fitness
    
    # --- Define fitness function for UNSCALED data ---    
    rf_eval_unscaled = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    )
    
    def fitness_function_unscaled(solution):
        selected_indices = np.where(solution == 1)[0]
        num_selected = len(selected_indices)
        
        if num_selected == 0:
            return 1.0  # Penalize solutions with no features
        
        # Select features from the datasets
        X_train_subset = X_train_unscaled[:, selected_indices]
        X_val_subset = X_val_unscaled[:, selected_indices]
        
        # Train Random Forest on the TRAINING subset
        rf_eval_unscaled.fit(X_train_subset, y_train)
        
        # Evaluate on VALIDATION set
        y_pred_val = rf_eval_unscaled.predict(X_val_subset)
        accuracy = accuracy_score(y_val, y_pred_val)
        
        # Calculate fitness value (minimize this)
        fitness = 0.99 * (1 - accuracy) + 0.01 * (num_selected / num_features_total)
        return fitness
    
    # --- Setup problem dictionaries for both scaled and unscaled data ---
    problem_dict_scaled = {
        "obj_func": fitness_function_scaled,
        "bounds": binary_bounds,
        "minmax": "min",
        "log_to": "file",
        "log_file": "sca_rf_scaled_convergence.log",
        "verbose": True
    }
    
    problem_dict_unscaled = {
        "obj_func": fitness_function_unscaled,
        "bounds": binary_bounds,
        "minmax": "min",
        "log_to": "file",
        "log_file": "sca_rf_unscaled_convergence.log",
        "verbose": True
    }
      # --- Run EO on SCALED data ---
    print("\n" + "="*50)
    print("Running EO with Random Forest on SCALED data...")
    print("="*50)
    
    start_time = time.time()
    model_eo_scaled = EO.OriginalEO(epoch=epoch, pop_size=pop_size)
    best_solution_scaled = model_eo_scaled.solve(problem_dict_scaled)
    end_time = time.time()
    print(f"EO on scaled data finished in {end_time - start_time:.2f} seconds.")
    
    # Extract selected indices from the best solution
    best_position_scaled = best_solution_scaled.solution
    selected_feature_indices_scaled = np.where(best_position_scaled == 1)[0]
    num_selected_features_scaled = len(selected_feature_indices_scaled)
    best_fitness_scaled = best_solution_scaled.target.fitness
    
    print(f"Selected {num_selected_features_scaled} features out of {num_features_total} (EO-RF - SCALED).")
    print(f"Best fitness found (EO-RF - SCALED): {best_fitness_scaled:.4f}")
      # --- Run EO on UNSCALED data ---
    print("\n" + "="*50)
    print("Running EO with Random Forest on UNSCALED data...")
    print("="*50)
    
    start_time = time.time()
    model_eo_unscaled = EO.OriginalEO(epoch=epoch, pop_size=pop_size)
    best_solution_unscaled = model_eo_unscaled.solve(problem_dict_unscaled)
    end_time = time.time()
    print(f"EO on unscaled data finished in {end_time - start_time:.2f} seconds.")
    
    # Extract selected indices from the best solution
    best_position_unscaled = best_solution_unscaled.solution
    selected_feature_indices_unscaled = np.where(best_position_unscaled == 1)[0]
    num_selected_features_unscaled = len(selected_feature_indices_unscaled)
    best_fitness_unscaled = best_solution_unscaled.target.fitness
    
    print(f"Selected {num_selected_features_unscaled} features out of {num_features_total} (EO-RF - UNSCALED).")
    print(f"Best fitness found (EO-RF - UNSCALED): {best_fitness_unscaled:.4f}")
      # --- Compare results ---
    print("\n" + "="*50)
    print("COMPARISON OF EO WITH RANDOM FOREST RESULTS")
    print("="*50)
    print(f"Scaled features selected: {num_selected_features_scaled}")
    print(f"Unscaled features selected: {num_selected_features_unscaled}")
    print(f"Best fitness (scaled): {best_fitness_scaled:.4f}")
    print(f"Best fitness (unscaled): {best_fitness_unscaled:.4f}")
    
    # Calculate overlap between selected features
    overlap_indices = np.intersect1d(selected_feature_indices_scaled, selected_feature_indices_unscaled)
    overlap_percentage = len(overlap_indices) / max(len(selected_feature_indices_scaled), len(selected_feature_indices_unscaled)) * 100
    
    print(f"Feature overlap: {len(overlap_indices)} features ({overlap_percentage:.1f}%)")
      # --- Get convergence history ---
    try:
        eo_history_scaled = np.array(model_eo_scaled.history.list_global_best_fit)
        eo_history_unscaled = np.array(model_eo_unscaled.history.list_global_best_fit)
        
        # Plot convergence comparison
        plt.figure(figsize=(10, 6))
        plt.plot(eo_history_scaled, 'b-', label='Scaled Features (EO-RF)')
        plt.plot(eo_history_unscaled, 'r-', label='Unscaled Features (EO-RF)')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value (lower is better)')
        plt.title('EO-RF Convergence: Scaled vs Unscaled Features')
        plt.legend()
        plt.grid(True)
        plt.savefig('eo_rf_convergence_comparison.png')
        plt.show()
    except Exception as e:
        print(f"Could not extract or plot convergence history (EO-RF): {e}")
        eo_history_scaled = np.array([])
        eo_history_unscaled = np.array([])
      # --- Save results to cache ---
    print("\nSaving feature selection results to cache (EO-RF)...")
    np.save(INDICES_CACHE_PATH_SCALED, selected_feature_indices_scaled)
    np.save(INDICES_CACHE_PATH_UNSCALED, selected_feature_indices_unscaled)
    np.save(EO_HISTORY_SCALED_PATH, eo_history_scaled)
    np.save(EO_HISTORY_UNSCALED_PATH, eo_history_unscaled)
    
    # Use scaled indices for backward compatibility
    selected_feature_indices = selected_feature_indices_scaled
    np.save(INDICES_CACHE_PATH, selected_feature_indices)
    
    print("Feature selection results cached successfully (EO-RF).")


# # 6. Classification using Random Forest with Selected Features

if not SKIP_TRAINING:
    # Select the features chosen by SCA
    if 'selected_feature_indices_scaled' not in locals() or 'selected_feature_indices_unscaled' not in locals():
        print("Error: Feature indices not found. Cannot train model.")
        if os.path.exists(INDICES_CACHE_PATH_SCALED) and os.path.exists(INDICES_CACHE_PATH_UNSCALED):
            print(f"Attempting to load indices from cache...")
            selected_feature_indices_scaled = np.load(INDICES_CACHE_PATH_SCALED)
            selected_feature_indices_unscaled = np.load(INDICES_CACHE_PATH_UNSCALED)
            print(f"Loaded {len(selected_feature_indices_scaled)} scaled indices and {len(selected_feature_indices_unscaled)} unscaled indices.")
        else:
            print("Cache files not found either. Exiting.")
            exit()

    X_train_selected_scaled = X_train_scaled[:, selected_feature_indices_scaled]
    X_val_selected_scaled = X_val_scaled[:, selected_feature_indices_scaled]
    X_test_selected_scaled = X_test_scaled[:, selected_feature_indices_scaled]
    
    X_train_selected_unscaled = X_train_unscaled[:, selected_feature_indices_unscaled]
    X_val_selected_unscaled = X_val_unscaled[:, selected_feature_indices_unscaled]
    X_test_selected_unscaled = X_test_unscaled[:, selected_feature_indices_unscaled]
    
    print("\n" + "="*50)
    print("COMPARING SCALED AND UNSCALED FEATURES WITH RANDOM FOREST")
    print("="*50)
    
    # Basic Random Forest models for comparison
    rf_scaled = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_unscaled = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    print("Training initial Random Forest models...")
    rf_scaled.fit(X_train_selected_scaled, y_train)
    rf_unscaled.fit(X_train_selected_unscaled, y_train)
    
    print("Evaluating on validation set...")
    y_pred_val_scaled = rf_scaled.predict(X_val_selected_scaled)
    y_pred_val_unscaled = rf_unscaled.predict(X_val_selected_unscaled)
    
    accuracy_scaled = accuracy_score(y_val, y_pred_val_scaled)
    accuracy_unscaled = accuracy_score(y_val, y_pred_val_unscaled)
    
    print(f"Initial validation accuracy with scaled features: {accuracy_scaled:.4f}")
    print(f"Initial validation accuracy with unscaled features: {accuracy_unscaled:.4f}")
    
    # Save the initial models as a baseline to compare against
    initial_rf_scaled = rf_scaled
    initial_rf_unscaled = rf_unscaled
    
    # Determine which feature set is better
    if accuracy_scaled >= accuracy_unscaled:
        print("\nScaled features perform better. Using scaled features for Random Forest tuning.")
        use_scaled = True
        X_train_selected = X_train_selected_scaled
        X_val_selected = X_val_selected_scaled
        X_test_selected = X_test_selected_scaled
        selected_feature_indices = selected_feature_indices_scaled
        initial_rf = initial_rf_scaled
        initial_accuracy = accuracy_scaled
    else:
        print("\nUnscaled features perform better. Using unscaled features for Random Forest tuning.")
        use_scaled = False
        X_train_selected = X_train_selected_unscaled
        X_val_selected = X_val_selected_unscaled
        X_test_selected = X_test_selected_unscaled
        selected_feature_indices = selected_feature_indices_unscaled
        initial_rf = initial_rf_unscaled
        initial_accuracy = accuracy_unscaled
    
    print("\n" + "="*50)
    print("PERFORMING RANDOM FOREST HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
    print("="*50)
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Create the Random Forest classifier
    rf_base = RandomForestClassifier(random_state=42)
    
    print("Starting GridSearchCV (this may take a while)...")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    # Fit GridSearchCV on the training data
    grid_search.fit(X_train_selected, y_train)
    
    end_time = time.time()
    print(f"GridSearchCV finished in {end_time - start_time:.2f} seconds.")
    
    # Get best parameters and score
    best_params = grid_search.best_params_
    print(f"\nBest parameters found: {best_params}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Create a table of the top 5 parameter combinations
    cv_results = pd.DataFrame(grid_search.cv_results_)
    top_results = cv_results.sort_values('rank_test_score').head(5)
    
    print("\nTop 5 Parameter Combinations:")
    for i, row in top_results.iterrows():
        params = row['params']
        score = row['mean_test_score']
        std = row['std_test_score']
        print(f"- Score: {score:.4f} (±{std:.4f}) | {params}")
    
    # Get the best model from GridSearchCV
    rf_final = grid_search.best_estimator_
    
    # Evaluate the best model on the validation set
    print("\nEvaluating best model on validation set...")
    y_pred_val = rf_final.predict(X_val_selected)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Validation accuracy with best model: {val_accuracy:.4f}")
    
    # Evaluate the best model on the test set
    print("\nEvaluating best model on test set...")
    start_time = time.time()
    y_pred_test = rf_final.predict(X_test_selected)
    end_time = time.time()
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test accuracy with best model: {test_accuracy:.4f}")
    print(f"Prediction time: {end_time - start_time:.2f} seconds")
    
    # Compare with initial model accuracy
    print(f"Initial model validation accuracy: {initial_accuracy:.4f}")
    
    # If the GridSearchCV model performs worse than the initial model, use the initial model instead
    if val_accuracy < initial_accuracy:
        print("\n*** WARNING: GridSearchCV model performs worse than initial model! ***")
        print(f"Reverting to initial model with accuracy {initial_accuracy:.4f}")
        
        rf_final = initial_rf
        y_pred_test = rf_final.predict(X_test_selected)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"Using initial model with parameters: {initial_rf.get_params()}")
        best_params = initial_rf.get_params()
    
    # Log which features were used
    feature_type = "scaled" if use_scaled else "unscaled"
    print(f"Final model uses {feature_type} features with hyperparameters: {best_params}")
    
    # Save the final model and which feature set was used
    joblib.dump(rf_final, MODEL_CACHE_PATH)
    joblib.dump(scaler, SCALER_CACHE_PATH)
    # Save the selected indices we used (scaled or unscaled)
    np.save(INDICES_CACHE_PATH, selected_feature_indices)
    # Also save which type of features we used (scaled or unscaled)
    with open('feature_type_rf.txt', 'w') as f:
        f.write(feature_type)
    # Save the best hyperparameters for later reference
    joblib.dump(best_params, 'best_rf_params.joblib')
    
    print(f"Saved final Random Forest model to {MODEL_CACHE_PATH}")
    print(f"Saved feature type information to feature_type_rf.txt")
    print(f"Saved best hyperparameters to best_rf_params.joblib")
else:
    # When loading cached model, try to determine which features to use
    if os.path.exists('feature_type_rf.txt'):
        with open('feature_type_rf.txt', 'r') as f:
            feature_type = f.read().strip()
        print(f"Using {feature_type} features as specified in feature_type_rf.txt")
        
        # Load the appropriate indices
        if feature_type == "scaled":
            if os.path.exists(INDICES_CACHE_PATH_SCALED):
                selected_feature_indices = np.load(INDICES_CACHE_PATH_SCALED)
                X_val_selected = X_val_scaled[:, selected_feature_indices]
                X_test_selected = X_test_scaled[:, selected_feature_indices]
            else:
                # Fallback to generic indices if needed
                X_val_selected = X_val_scaled[:, selected_feature_indices]
                X_test_selected = X_test_scaled[:, selected_feature_indices]
        else:  # unscaled
            if os.path.exists(INDICES_CACHE_PATH_UNSCALED):
                selected_feature_indices = np.load(INDICES_CACHE_PATH_UNSCALED)
                X_val_selected = X_val_unscaled[:, selected_feature_indices]
                X_test_selected = X_test_unscaled[:, selected_feature_indices]
            else:
                # Fallback to generic indices but with unscaled data
                X_val_selected = X_val_unscaled[:, selected_feature_indices]
                X_test_selected = X_test_unscaled[:, selected_feature_indices]
    else:
        # Default to scaled features for backward compatibility
        print("No feature type information found. Defaulting to scaled features.")
        X_val_selected = X_val_scaled[:, selected_feature_indices]
        X_test_selected = X_test_scaled[:, selected_feature_indices]
    
    # Load best hyperparameters if available (for display purposes)
    if os.path.exists('best_rf_params.joblib'):
        best_params = joblib.load('best_rf_params.joblib')
        print(f"Best hyperparameters from GridSearchCV: {best_params}")
    
    print("Predicting on test set using cached model...")
    start_time = time.time()
    y_pred_test = rf_final.predict(X_test_selected)
    end_time = time.time()
    print(f"Prediction finished in {end_time - start_time:.2f} seconds.")


# # 7. Evaluation and Visualization

def evaluate_and_save_results():
    """Evaluate the Random Forest model and save comprehensive metrics."""
    print("\n\n" + "="*50)
    print("GENERATING COMPREHENSIVE EVALUATION METRICS")
    print("="*50)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    test_precision = precision_score(y_test, y_pred_test, average='weighted')
    test_recall = recall_score(y_test, y_pred_test, average='weighted')
    
    # Calculate validation metrics if validation set is available
    val_accuracy = None
    val_f1 = None
    if 'X_val_selected' in locals() and 'y_val' in locals():
        y_pred_val = rf_final.predict(X_val_selected)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        val_f1 = f1_score(y_val, y_pred_val, average='weighted')
    
    unique_emotions = sorted(np.unique(np.concatenate([y_test, y_pred_test])))
    emotion_names = [EMOTIONS.get(e, f"Unknown-{e}") for e in unique_emotions]
    
    # Confusion matrix with proper labels
    cm = confusion_matrix(y_test, y_pred_test, labels=unique_emotions)
    
    # Full classification report
    report = classification_report(
        y_test, 
        y_pred_test, 
        labels=unique_emotions,
        target_names=emotion_names,
        output_dict=True
    )
    
    # Get predictions for training set
    X_train_selected = X_train_scaled[:, selected_feature_indices] if use_scaled else X_train_unscaled[:, selected_feature_indices]
    y_train_pred = rf_final.predict(X_train_selected)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    
    # Compute per-class metrics
    train_per_class = {}
    val_per_class = {}
    test_per_class = {}
    
    for emotion in unique_emotions:
        # Get indices for this emotion
        train_indices = np.where(y_train == emotion)[0]
        test_indices = np.where(y_test == emotion)[0]
        
        # Calculate accuracies for this emotion
        if len(train_indices) > 0:
            train_emotion_true = y_train[train_indices]
            train_emotion_pred = y_train_pred[train_indices]
            train_per_class[emotion] = accuracy_score(train_emotion_true, train_emotion_pred)
        else:
            train_per_class[emotion] = 0
            
        if 'X_val_selected' in locals() and 'y_val' in locals():
            val_indices = np.where(y_val == emotion)[0]
            if len(val_indices) > 0:
                val_emotion_true = y_val[val_indices]
                val_emotion_pred = y_pred_val[val_indices]
                val_per_class[emotion] = accuracy_score(val_emotion_true, val_emotion_pred)
            else:
                val_per_class[emotion] = 0
        
        if len(test_indices) > 0:
            test_emotion_true = y_test[test_indices]
            test_emotion_pred = y_pred_test[test_indices]
            test_per_class[emotion] = accuracy_score(test_emotion_true, test_emotion_pred)
        else:
            test_per_class[emotion] = 0
    
    # Create confusion matrix figure
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=emotion_names,
        yticklabels=emotion_names
    )
    plt.title('Confusion Matrix (Random Forest)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_rf.png')
    
    # Create accuracy by emotion figure
    fig_acc, ax_acc = plt.subplots(figsize=(12, 6))
    
    # Convert per-class metrics to a format suitable for plotting
    emotion_dict = {}
    for emotion in unique_emotions:
        emotion_name = EMOTIONS.get(emotion, f"Unknown-{e}")
        train_acc = train_per_class.get(emotion, 0)
        val_acc = val_per_class.get(emotion, 0) if val_per_class else None
        test_acc = test_per_class.get(emotion, 0)
        
        data_dict = {'Train': train_acc, 'Test': test_acc}
        if val_acc is not None:
            data_dict['Validation'] = val_acc
            
        emotion_dict[emotion_name] = data_dict
    
    # Create DataFrame and plot
    emotion_df = pd.DataFrame(emotion_dict).T
    emotion_df.plot(kind='bar', ax=ax_acc)
    ax_acc.set_ylim(0, 1.1)
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_xlabel('Emotion')
    ax_acc.set_title('Accuracy by Emotion (Random Forest)')
    ax_acc.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig('accuracy_by_emotion_rf.png')
    
    # Create feature importance chart
    fig_imp = None
    try:
        # Get feature importances from Random Forest
        importances = rf_final.feature_importances_
        
        # Get top N most important features
        top_n = min(20, len(importances))
        top_indices = np.argsort(importances)[-top_n:][::-1]
        top_importances = importances[top_indices]
        
        # Map back to original feature indices
        original_feature_indices = [selected_feature_indices[i] for i in top_indices]
        
        # Create importance figure
        fig_imp, ax_imp = plt.subplots(figsize=(12, 6))
        ax_imp.bar(range(top_n), top_importances)
        ax_imp.set_xticks(range(top_n))
        ax_imp.set_xticklabels([f"Feature {i}" for i in original_feature_indices], rotation=90)
        ax_imp.set_title('Top Feature Importances (Random Forest)')
        ax_imp.set_ylabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('feature_importance_rf.png')
    except Exception as e:
        print(f"Could not create feature importance figure: {e}")

    # Create model report
    report_content = f"""
    # Arabic Emotion Recognition Model Evaluation Report (Random Forest)

    ## Overall Performance with {feature_type.capitalize()} Features
    - Training Accuracy: {train_accuracy:.4f}
    - Validation Accuracy: {val_accuracy:.4f if val_accuracy else 'N/A'}
    - Test Accuracy: {test_accuracy:.4f}
    - Test F1 Score (Weighted): {test_f1:.4f}
    - Test Precision (Weighted): {test_precision:.4f}
    - Test Recall (Weighted): {test_recall:.4f}

    ## Model Parameters
    {best_params}

    ## Selected Features
    - Number of selected features: {len(selected_feature_indices)} out of {X_train_scaled.shape[1]}
    - Feature selection method: EO (Equilibrium Optimizer)
    """

    with open('evaluation_report_rf.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # Save evaluation results
    results = {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'confusion_matrix': cm,
        'unique_emotions': unique_emotions,
        'emotion_names': emotion_names,
        'train_size': len(y_train),
        'val_size': len(y_val) if 'y_val' in locals() else 0,
        'test_size': len(y_test),
        'classification_report': report,
        'train_per_class': train_per_class,
        'val_per_class': val_per_class if 'X_val_selected' in locals() else {},
        'test_per_class': test_per_class,
        'confusion_matrix_fig': fig_cm,
        'accuracy_by_emotion_fig': fig_acc,
        'feature_importance_fig': fig_imp,
        'best_params': best_params
    }
    
    joblib.dump(results, EVALUATION_RESULTS_PATH)
    print(f"Evaluation results saved to '{EVALUATION_RESULTS_PATH}'")
    
    # Print evaluation summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY (RANDOM FOREST)")
    print("="*50)
    print(f"Training Set: {len(y_train)} samples")
    print(f"Validation Set: {len(y_val)} samples" if 'y_val' in locals() else "Validation Set: N/A")
    print(f"Test Set: {len(y_test)} samples")
    print(f"Selected Features: {len(selected_feature_indices)} out of {X_train_scaled.shape[1]}")
    print("="*50)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}" if val_accuracy else "Validation Accuracy: N/A")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("="*50)
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print("="*50)
    
    return results


# Run evaluation if the model is available
if 'rf_final' in locals() and 'y_pred_test' in locals():
    evaluate_and_save_results()


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arabic Emotion Recognition with EO and Random Forest')
    parser.add_argument('--clear-all', action='store_true', 
                        help='Clear all cached files before running')
    parser.add_argument('--clear-model', action='store_true', 
                        help='Clear only model files before running')
    parser.add_argument('--clear-evaluation', action='store_true', 
                        help='Clear only evaluation results before running')
    args = parser.parse_args()
    
    # Handle cache clearing if requested
    if args.clear_all:
        cache_files = [
            INDICES_CACHE_PATH_SCALED, INDICES_CACHE_PATH_UNSCALED, INDICES_CACHE_PATH,
            SCALER_CACHE_PATH, MODEL_CACHE_PATH, EVALUATION_RESULTS_PATH,
            EO_HISTORY_SCALED_PATH, EO_HISTORY_UNSCALED_PATH
        ]
        for file in cache_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Deleted cache file: {file}")
    
    elif args.clear_model:
        model_files = [
            INDICES_CACHE_PATH_SCALED, INDICES_CACHE_PATH_UNSCALED, INDICES_CACHE_PATH,
            SCALER_CACHE_PATH, MODEL_CACHE_PATH, EVALUATION_RESULTS_PATH
        ]
        for file in model_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Deleted model file: {file}")
    
    elif args.clear_evaluation:
        if os.path.exists(EVALUATION_RESULTS_PATH):
            os.remove(EVALUATION_RESULTS_PATH)
            print(f"Deleted evaluation results: {EVALUATION_RESULTS_PATH}")
    
    print("Random Forest training and evaluation complete.")
