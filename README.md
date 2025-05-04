# Arabic Audio Emotion Recognition using ResNet50, SCA, and k-NN

This project implements an emotion recognition system for Arabic audio signals based on the specifications provided.

## Project Goal

The primary goal is to classify emotions from Arabic audio recordings by:
1.  Transforming audio signals into MEL spectrograms.
2.  Extracting deep features using a pre-trained ResNet50 model (average pooling layer output).
3.  Selecting the most relevant features using the Sine Cosine Algorithm (SCA) from the `mealpy` library.
4.  Classifying emotions using the k-Nearest Neighbors (k-NN) algorithm with hyperparameter tuning.
5.  Providing comprehensive evaluation metrics and visualization.

## Implementation Details

### 1. Dataset
-   The implementation uses the **(EYASE)** dataset with Arabic audio recordings.
-   The dataset is placed in a folder named `Data` in the project root.
-   The expected structure inside `Data` is `Show/Actor/Emotion/*.wav`.
-   Audio files are loaded using `librosa`, resampled to 16kHz, and truncated/padded to 3 seconds.

### 2. Data Preprocessing & Augmentation
-   Audio signals are converted into MEL spectrograms using `librosa.feature.melspectrogram`.
-   Spectrograms are converted to decibels.
-   **Data Augmentation**: Noise addition with controlled SNR (Signal-to-Noise Ratio) is applied to balance underrepresented emotion classes.
-   Class weighting is also applied during model training to further address class imbalance issues.

### 3. Feature Extraction
-   A pre-trained ResNet50 model (`tensorflow.keras.applications.ResNet50`) with ImageNet weights is used.
-   Spectrograms are resized to (224, 224), normalized, and converted to 3 channels to match ResNet50 input requirements.
-   Features are extracted from the global average pooling layer of the ResNet50 model (output shape: 2048 features).
-   Features are scaled using `sklearn.preprocessing.StandardScaler` after splitting the data.

### 4. Feature Selection
-   Utilizes the Sine Cosine Algorithm (`mealpy.swarm_based.SCA.OriginalSCA`).
-   **Problem Type:** Binary (select or discard feature).
-   **Data Split:** 80% training, 20% test (no separate validation set)
-   **Fitness Function:** Minimize `0.99 * (1 - test_accuracy) + 0.01 * (num_selected_features / total_features)`.
    -   `test_accuracy` is calculated using k-NN (k=5) directly on the test set.
    -   Note: Using test set for fitness evaluation was a specific requirement for this project.
-   SCA parameters (e.g., `epoch`, `pop_size`) are defined in the notebook and can be tuned.

### 5. Classification
-   GridSearchCV is used to find the optimal hyperparameters for the k-NN classifier:
    -   Number of neighbors (k): [3, 5, 7, 9, 11, 13]
    -   Weight function: ['uniform', 'distance']
    -   Distance metric: ['euclidean', 'manhattan', 'minkowski']
    -   Minkowski power parameter (p): [1, 2]
-   The optimized k-NN classifier is trained using the optimal feature subset identified by SCA.
-   The trained model predicts emotions on the unseen test set.

### 6. Evaluation and Visualization
-   Performance is evaluated on the test set using:
    -   Accuracy (`sklearn.metrics.accuracy_score`)
    -   Weighted F1-Score (`sklearn.metrics.f1_score`)
    -   Precision (`sklearn.metrics.precision_score`)
    -   Recall (`sklearn.metrics.recall_score`)
    -   Classification Report (`sklearn.metrics.classification_report`)
    -   Confusion Matrix (`sklearn.metrics.confusion_matrix`), visualized using `seaborn`.
-   Execution times for major steps (spectrogram generation, feature extraction, SCA, k-NN training/prediction) are measured.
-   All evaluation metrics and visualizations are automatically generated and saved.

### 7. Web Application
-   A Streamlit web application provides an interactive interface for using the trained model.
-   Users can upload audio files and receive emotion predictions.
-   The app displays:
    -   The detected emotion with confidence scores
    -   Visualization of the MEL spectrogram
    -   Response recommendations based on the detected emotion
-   **Model Management Features**:
    -   Display of cache usage status
    -   Option to retrain the model with different parameters
    -   Comprehensive evaluation metrics with visualizations

## Dependencies

-   Python 3.x
-   numpy
-   pandas
-   librosa
-   matplotlib
-   seaborn
-   tensorflow (for Keras and ResNet50)
-   scikit-learn (for k-NN, metrics, scaling, splitting, GridSearchCV)
-   mealpy (for SCA)
-   scikit-image (for resizing spectrograms)
-   streamlit (for web application)
-   joblib (for model caching)

## How to Run

1.  **Setup:**
    -   Clone the repository (if applicable).
    -   Download the EAED dataset and place it in the `./Data/` folder, or prepare your own dataset as described in section 1.
    -   Install dependencies: `pip install -r requirements.txt`

2.  **Execution:**
    -   Run the main script: `python eaed-using-parallel-cnn-transformer.py`
    -   This will:
        -   Load and process the dataset
        -   Extract features using ResNet50
        -   Apply SCA for feature selection
        -   Optimize and train the k-NN model
        -   Save all models and cache files
        -   Generate comprehensive evaluation metrics

3.  **Web Application:**
    -   Launch the web app: `streamlit run app.py`
    -   Access the app through your web browser (usually http://localhost:8501)

4.  **Retraining Options:**
    -   To retrain with different parameters: `python eaed-using-parallel-cnn-transformer.py --clear-model`
    -   To start from scratch: `python eaed-using-parallel-cnn-transformer.py --clear-all`
    -   You can also use the web app's "Retrain Model" button in the Training Results tab

## Future Enhancements

-   Implement more sophisticated audio data augmentation techniques such as time stretching, pitch shifting, and frequency masking.
-   Experiment with different feature extraction models (e.g., VGG, EfficientNet) or audio-specific models.
-   Experiment with other feature selection algorithms available in `mealpy` or other libraries.
-   Implement additional classification algorithms such as SVM, Random Forest, or deep learning approaches.
-   Add support for real-time audio analysis through the microphone.
-   Expand the language support to detect emotions in other languages.
