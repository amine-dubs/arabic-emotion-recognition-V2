# How to Retrain the Model

This document explains how to retrain the Arabic emotion detection model with updated parameters and approaches.

## Overview of the Current Implementation

The current implementation follows these key principles as specified in `prompt.txt`:

1. Audio signals are converted to mel spectrograms
2. Deep features are extracted using ResNet50
3. **SCA (Sine Cosine Algorithm) is used for feature selection**
4. **The fitness function uses the test set directly for evaluation** (as requested in the prompt)
5. The best k-NN model from SCA is used for emotion classification in the app
6. Evaluations include both scaled and unscaled versions of the features

## Retraining Steps

### Option 1: Using the Web App Interface

The easiest way to retrain is through the app interface:

1. Run the Streamlit app: `streamlit run app.py`
2. Navigate to the "Training Results" tab
3. Expand the "Retrain options" section
4. Choose from:
   - **Quick retrain**: Only retrains the k-NN model (fastest)
   - **Full retune**: Runs feature selection with SCA and model training
   - **Complete retraining**: Starts from scratch with all processing steps
5. Click "Retrain Model" button
6. Wait for the process to complete (may take several minutes)
7. Refresh the app to see updated results

### Option 2: Using Command Line

For more control over the retraining process:

#### Clear Existing Cache Files

First, decide what cache files to clear:

```bash
# Clear everything for a complete retrain
python clear_cache.py --all

# Clear only model and evaluation results
python clear_cache.py --model --evaluation

# Clear selected indices and model
python clear_cache.py --indices --model
```

#### Run the Training Script

```bash
# Basic run with default parameters
python eaed-using-parallel-cnn-transformer.py

# Run with specific command line arguments
python eaed-using-parallel-cnn-transformer.py --clear-model
```

## Important Notes on the Training Approach

### Modified Fitness Function

As requested, the fitness function now uses the test set directly for evaluation during SCA feature selection:

```python
def fitness_function(solution):
    selected_indices = np.where(solution == 1)[0]
    num_selected = len(selected_indices)

    if num_selected == 0:
        return 1.0  # Penalize solutions with no features

    # Select features from TRAINING and TEST sets
    X_train_subset = X_train_scaled[:, selected_indices]
    X_test_subset = X_test_scaled[:, selected_indices]  # Use TEST set for evaluation

    # Train k-NN on the TRAINING subset
    knn_eval.fit(X_train_subset, y_train)

    # Evaluate on TEST set
    y_pred_test = knn_eval.predict(X_test_subset)  # Predict on TEST set
    accuracy = accuracy_score(y_test, y_pred_test)  # Calculate accuracy using TEST set

    # Calculate fitness value (minimize this)
    fitness = 0.99 * (1 - accuracy) + 0.01 * (num_selected / num_features_total)
    return fitness
```

This means the SCA algorithm optimizes feature selection directly against the test set performance. While this approach was specifically requested and follows the principles in `prompt.txt`, it's important to note this could potentially lead to optimistically biased evaluation results.

### Using the Best k-NN Model

The training process now simplifies the final k-NN model training to match the model used in the SCA fitness function. This ensures consistency between the optimization process and the final deployed model.

### Evaluation Without Normalization (Scaling)

The code now includes evaluation of the model's performance both with and without feature scaling, allowing for comparison between these two approaches. The unscaled evaluation results are included in the final assessment.

## Evaluating Model Performance

After training, review the model's performance:

1. Check the `evaluation_results.joblib` file or the Training Results tab in the app
2. Compare the results with and without scaling
3. Examine the confusion matrices to identify any patterns in misclassifications
4. Review the selected features to understand what the model considers important

## Future Improvements

- Implement cross-validation within SCA to reduce potential overfitting
- Add more emotion classes if additional data becomes available
- Experiment with different feature extraction techniques beyond ResNet50
- Add real-time audio classification capabilities to the app