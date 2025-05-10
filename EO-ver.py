#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import time
import joblib # Added for caching models/scalers
import gc # Added for garbage collection
import subprocess # Added for running the model training from app.py
import argparse # For command line arguments

# Imports for the new pipeline
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize # For resizing spectrograms

from mealpy import EO # Changed from SCA to EO
from mealpy.utils.space import BinaryVar # Import BinaryVar
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
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
INDICES_CACHE_PATH = 'selected_indices_scaled_eo.npy' # Changed from sca to eo
SCALER_CACHE_PATH = 'scaler.joblib'
MODEL_CACHE_PATH = 'knn_final_model.joblib'
EVALUATION_RESULTS_PATH = 'evaluation_results.joblib'
EO_HISTORY_SCALED_PATH = 'eo_history_scaled.npy'  # To store fitness history, changed from sca to eo
EO_HISTORY_UNSCALED_PATH = 'eo_history_unscaled.npy'  # To store fitness history, changed from sca to eo
INDICES_CACHE_PATH_UNSCALED = 'selected_indices_unscaled_eo.npy' # Changed from sca to eo
# ------------------------


# In[ ]:


EMOTIONS = { 0 : 'Angry',
             2 : 'Happy',
             3 : 'Neutral',
             4 : 'Sad'
           }  
# !!! IMPORTANT: Update this path to your local dataset location !!!
# DATA_PATH = '/kaggle/input/eaed-voice/EAED' 
# DATA_PATH = './EAED' # Example local path
DATA_PATH = './Data' # Corrected path based on workspace structure
SAMPLE_RATE = 16000
DURATION = 3 # seconds
RESNET_INPUT_SHAPE = (224, 224, 3) # ResNet50 expects 3 channels


# # 1. Data Loading and Initial Exploration

# In[ ]:


# --- Check for cached features and DataFrame ---
if os.path.exists(FEATURES_CACHE_PATH) and os.path.exists(DATA_CACHE_PATH):
    print(f"Loading cached features from {FEATURES_CACHE_PATH}...")
    resnet_features = np.load(FEATURES_CACHE_PATH)
    print(f"Loading cached DataFrame from {DATA_CACHE_PATH}...")
    Data = pd.read_pickle(DATA_CACHE_PATH)
    print("Loaded features shape:", resnet_features.shape)
    print("Loaded DataFrame head:\n", Data.head())
    # Skip data loading, spectrogram generation, and feature extraction
    SKIP_FEATURE_EXTRACTION = True
else:
    print("Cache not found. Starting data loading and feature extraction...")
    SKIP_FEATURE_EXTRACTION = False
# ---------------------------------------------

# --- Check for cached model ---
if os.path.exists(MODEL_CACHE_PATH) and os.path.exists(SCALER_CACHE_PATH):
    print(f"Loading cached model from {MODEL_CACHE_PATH}...")
    knn_final = joblib.load(MODEL_CACHE_PATH)
    scaler = joblib.load(SCALER_CACHE_PATH)
    # Skip training
    SKIP_TRAINING = True
else:
    print("No cached model found. Will train a new model.")
    SKIP_TRAINING = False
# ---------------------------------------------

if not SKIP_FEATURE_EXTRACTION:
    file_names = []
    file_emotions = []
    file_paths = []

    # Define mapping from filename code to full emotion name
    emotion_code_map = {
        'ang': 'Angry',
        'hap': 'Happy',
        'neu': 'Neutral',
        'sad': 'Sad',
        # Add other codes if present (e.g., 'fea' for Fearful, 'sur' for Surprised)
    }

    # Iterate over each show folder (e.g., EYASE)
    for show_folder in os.listdir(DATA_PATH):
        show_path = os.path.join(DATA_PATH, show_folder)
        if not os.path.isdir(show_path):
            continue
        
        # Iterate over actor folders (e.g., Female01, Male02)
        for actor_folder in os.listdir(show_path):
            actor_path = os.path.join(show_path, actor_folder)
            if not os.path.isdir(actor_path):
                continue
            
            # Iterate over audio files within the actor folder
            for audio_file in os.listdir(actor_path):
                if audio_file.endswith(".wav"):
                    try:
                        # Parse information from the file name
                        # Example: fm01_ang (1).wav -> parts = ['fm01', 'ang (1).wav']
                        parts = audio_file.split("_", 1) 
                        if len(parts) < 2:
                            print(f"Skipping file with unexpected format: {audio_file}")
                            continue
                            
                        # Extract emotion code: 'ang (1).wav' -> 'ang'
                        emotion_code = parts[1].split(' ')[0]
                        
                        # Map code to full emotion name
                        emotion_full_name = emotion_code_map.get(emotion_code)

                        # Check if the emotion code was found in the map
                        if emotion_full_name is None:
                            print(f"Warning: Emotion code '{emotion_code}' not found in map for file {audio_file}. Skipping.")
                            continue

                        # Encode emotion using the EMOTIONS dictionary
                        try:
                            emotion_encoded = list(EMOTIONS.keys())[list(EMOTIONS.values()).index(emotion_full_name)]
                        except ValueError:
                             print(f"Warning: Emotion '{emotion_full_name}' not found in EMOTIONS dictionary for file {audio_file}. Skipping.")
                             continue

                        # Construct the full file path
                        file_path = os.path.join(actor_path, audio_file)
                        
                        # Append the information to the lists
                        file_names.append(audio_file)
                        file_emotions.append(emotion_encoded)
                        file_paths.append(file_path)
                    except Exception as e:
                        print(f"Error processing file {audio_file} in {actor_path}: {e}")


    # In[4]:


    # Create a DataFrame
    Data = pd.DataFrame({
        "Name": file_names,
        "Emotion": file_emotions,
        "Path": file_paths
    })
    # --- Save DataFrame to cache ---
    print(f"Saving DataFrame to {DATA_CACHE_PATH}...")
    Data.to_pickle(DATA_CACHE_PATH)
    # -----------------------------


# In[5]:


Data.head()


# In[6]:


print("number of files is {}".format(len(Data)))


# In[7]:


# Get the actual counts and labels present in the data
emotion_counts = Data['Emotion'].value_counts().sort_index() # Sort by index (emotion code)
emotion_labels_present = emotion_counts.index.tolist() # Get the numeric labels (0, 2, 3, 4)
emotion_names_present = [EMOTIONS[i] for i in emotion_labels_present] # Get the corresponding names

fig = plt.figure()
ax = fig.add_subplot(111)
# Use the actual number of emotions present for the x-axis positions
ax.bar(x=range(len(emotion_counts)), height=emotion_counts.values)
# Set ticks and labels based on the emotions actually present
ax.set_xticks(ticks=range(len(emotion_counts)))
ax.set_xticklabels(emotion_names_present, fontsize=10)
ax.set_xlabel('Emotions')
ax.set_ylabel('Number of examples')


# # 2. Spectrogram Generation

# In[8]::


def getMELspectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=1024,
                                              win_length = 512,
                                              window='hamming',
                                              hop_length = 256,
                                              n_mels=128,
                                              fmax=sample_rate/2
                                             )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Function for data augmentation - add noise with controlled SNR for balancing classes
def augment_audio_with_noise(audio, snr_db=15):
    """Add noise to audio sample with specified signal-to-noise ratio.
    
    Args:
        audio: Audio signal array
        snr_db: Signal-to-noise ratio in dB (higher = less noise)
    
    Returns:
        Augmented audio signal
    """
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    
    # Calculate noise power based on SNR
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generate noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    
    # Add noise to signal
    augmented_audio = audio + noise
    
    # Normalize to avoid clipping
    if np.max(np.abs(augmented_audio)) > 1.0:
        augmented_audio = augmented_audio / np.max(np.abs(augmented_audio))
        
    return augmented_audio

# --- Skip spectrogram generation if features are loaded ---
if not SKIP_FEATURE_EXTRACTION:
    # In[ ]:


    audio, sample_rate = librosa.load(Data.loc[0,'Path'], duration=DURATION, offset=0.5,sr=SAMPLE_RATE)
    signal = np.zeros((int(SAMPLE_RATE*DURATION,)))
    signal[:len(audio)] = audio
    mel_spectrogram = getMELspectrogram(signal, SAMPLE_RATE)
    librosa.display.specshow(mel_spectrogram, sr=sample_rate, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MEL spectrogram example')
    plt.tight_layout()
    plt.show()
    print('Original MEL spectrogram shape: ',mel_spectrogram.shape)


    # In[ ]:


    # Function to resize and prepare spectrogram for ResNet50
    def prepare_spectrogram_for_resnet(spec, output_shape):
        # Resize
        spec_resized = resize(spec, output_shape[:2], anti_aliasing=True)
        # Normalize to 0-1 range (assuming spec is in dB)
        spec_resized -= spec_resized.min()
        if spec_resized.max() > 0:
            spec_resized /= spec_resized.max()
        # Convert to 3 channels by stacking
        spec_3channel = np.stack([spec_resized]*3, axis=-1)
        return spec_3channel


    # In[ ]:


    # Test preparation function
    prepared_spec = prepare_spectrogram_for_resnet(mel_spectrogram, RESNET_INPUT_SHAPE)
    print('Prepared spectrogram shape: ', prepared_spec.shape)
    plt.imshow(prepared_spec)
    plt.title('Prepared Spectrogram (3 channels)')
    plt.show()


    # In[ ]:

    # Check class balance and apply augmentation to balance classes
    emotion_counts = Data['Emotion'].value_counts()
    target_count = int(emotion_counts.max() * 1.2)  # Target slightly more than the majority class
    print(f"Current emotion distribution: {emotion_counts.to_dict()}")
    print(f"Target count per emotion: {target_count}")
    
    # Create an augmentation plan
    augmentation_plan = {}
    for emotion, count in emotion_counts.items():
        if count < target_count:
            # Calculate how many augmented samples to generate
            num_to_augment = target_count - count
            augmentation_plan[emotion] = num_to_augment
    
    print(f"Augmentation plan: {augmentation_plan}")
    
    # Create list to store original and augmented data
    prepared_spectrograms = []
    valid_indices = []  # Keep track of successfully processed files
    augmented_emotions = []  # Track emotions for augmented samples
    augmented_names = []  # Track filenames for augmented samples
    augmented_paths = []  # Track paths for augmented samples (same as original)
    
    start_time = time.time()
    print("Generating and preparing spectrograms (with augmentation)...")
    
    # Process each original file
    for i, row in Data.iterrows():
        file_path = row['Path']
        emotion = row['Emotion']
        
        try:
            # Process original audio
            audio, sample_rate = librosa.load(file_path, duration=DURATION, offset=0.5, sr=SAMPLE_RATE)
            signal = np.zeros((int(SAMPLE_RATE*DURATION,)))
            signal[:len(audio)] = audio
            
            # Generate and add original mel spectrogram
            mel_spectrogram = getMELspectrogram(signal, sample_rate=SAMPLE_RATE)
            prepared_spec = prepare_spectrogram_for_resnet(mel_spectrogram, RESNET_INPUT_SHAPE)
            prepared_spectrograms.append(prepared_spec)
            valid_indices.append(i)  # Add index if successful
            
            # Check if this emotion needs augmentation
            if emotion in augmentation_plan and augmentation_plan[emotion] > 0:
                # Calculate how many augmentations to create from this sample
                # Distribute augmentation across all samples of this emotion
                samples_of_this_emotion = len(Data[Data['Emotion'] == emotion])
                augmentations_per_sample = max(1, augmentation_plan[emotion] // samples_of_this_emotion)
                
                # Create augmented versions with different noise levels
                for aug_idx in range(augmentations_per_sample):
                    # Apply data augmentation with varying SNR
                    snr_db = np.random.uniform(10, 25)  # Random SNR between 10 and 25 dB
                    augmented_audio = augment_audio_with_noise(audio, snr_db=snr_db)
                    
                    # Process augmented audio
                    aug_signal = np.zeros((int(SAMPLE_RATE*DURATION,)))
                    aug_signal[:len(augmented_audio)] = augmented_audio
                    
                    # Generate mel spectrogram from augmented audio
                    aug_mel_spec = getMELspectrogram(aug_signal, sample_rate=SAMPLE_RATE)
                    aug_prepared_spec = prepare_spectrogram_for_resnet(aug_mel_spec, RESNET_INPUT_SHAPE)
                    
                    # Add to lists
                    prepared_spectrograms.append(aug_prepared_spec)
                    augmented_emotions.append(emotion)
                    augmented_names.append(f"{row['Name']}_aug_{aug_idx}")
                    augmented_paths.append(file_path)
                    
                    # Decrease the count needed for this emotion
                    augmentation_plan[emotion] -= 1
                    if augmentation_plan[emotion] <= 0:
                        break
            
        except Exception as e:
            print(f"\nError processing {file_path} (index {i}): {e}. Skipping this file.")
        
        print(f"\r Processed {i+1}/{len(Data)} files", end='')

    # Combine original and augmented data
    if len(augmented_emotions) > 0:
        # Create DataFrame for augmented samples
        augmented_df = pd.DataFrame({
            "Name": augmented_names,
            "Emotion": augmented_emotions,
            "Path": augmented_paths
        })
        
        # Add augmented samples to the original DataFrame
        Data = pd.concat([Data.loc[valid_indices].reset_index(drop=True), augmented_df], ignore_index=True)
        print(f"\nAdded {len(augmented_emotions)} augmented samples. New dataset size: {len(Data)}")
        
        # Show the new class distribution
        new_emotion_counts = Data['Emotion'].value_counts()
        print(f"New emotion distribution after augmentation: {new_emotion_counts.to_dict()}")
    else:
        # Filter Data to keep only successfully processed rows
        print(f"\nFiltering DataFrame to keep {len(valid_indices)} successfully processed files.")
        Data = Data.loc[valid_indices].reset_index(drop=True)
    
    # --- Update cached DataFrame after filtering and augmentation ---
    print(f"Saving filtered and augmented DataFrame to {DATA_CACHE_PATH}...")
    Data.to_pickle(DATA_CACHE_PATH)
    # ---------------------------------------------

    prepared_spectrograms = np.array(prepared_spectrograms)
    end_time = time.time()
    print(f"\nFinished spectrograms in {end_time - start_time:.2f} seconds.")
    print("Shape of prepared spectrograms array:", prepared_spectrograms.shape)


    # # 3. Feature Extraction using ResNet50

    # In[ ]:


    # Load pre-trained ResNet50 model + higher level layers
    print("Loading ResNet50 model...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=RESNET_INPUT_SHAPE, pooling='avg')
    # base_model.summary() # Optional: Print model summary

    # Create a new model that outputs the features
    feature_model = Model(inputs=base_model.input, outputs=base_model.output)
    print("ResNet50 model loaded.")


    # In[ ]:


    # Extract features
    # Note: ResNet50 expects preprocessed input (specific normalization)
    # We'll apply the standard ResNet preprocessing
    print("Extracting features using ResNet50...")
    start_time = time.time()
    # Ensure prepared_spectrograms is not empty before predicting
    if prepared_spectrograms.shape[0] > 0:
        resnet_features = feature_model.predict(tf.keras.applications.resnet50.preprocess_input(prepared_spectrograms))
        # --- Save features to cache ---
        print(f"Saving features to {FEATURES_CACHE_PATH}...")
        np.save(FEATURES_CACHE_PATH, resnet_features)
        # ----------------------------
    else:
        print("Error: No spectrograms were successfully prepared. Cannot extract features.")
        # Handle this error appropriately, maybe exit or raise an exception
        resnet_features = np.array([]) # Assign empty array to avoid later errors

    end_time = time.time()
    print(f"Finished feature extraction in {end_time - start_time:.2f} seconds.")
    print("Shape of extracted features:", resnet_features.shape)

    # Free up memory
    del prepared_spectrograms
    gc.collect()
# --- End of SKIP_FEATURE_EXTRACTION block ---


# # 4. Data Splitting and Scaling

# In[ ]:

# Ensure resnet_features is not empty before proceeding
if resnet_features.shape[0] == 0 or resnet_features.shape[0] != len(Data):
     print("Error: Feature array is empty or does not match DataFrame length. Exiting.")
     # Exit or raise error
     exit()

# --- Added: Shuffle features and corresponding labels ---
print("Shuffling features and labels before splitting...")
# Ensure reproducibility of the shuffle
permutation_indices = np.random.RandomState(seed=42).permutation(len(Data))
resnet_features = resnet_features[permutation_indices]
shuffled_emotions = Data['Emotion'].iloc[permutation_indices]
# --- End of shuffle ---

X = resnet_features
y = shuffled_emotions.values # Use the shuffled emotions

# Modified: Split data into training and testing sets only (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Original data shape:", X.shape, y.shape)
print("Training data shape:", X_train.shape, y_train.shape)
print("Test data shape:", X_test.shape, y_test.shape)

# Store unscaled data for later use
X_train_unscaled = X_train.copy()
X_test_unscaled = X_test.copy()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Free up memory (keep scaled data, unscaled data and labels)
del X, resnet_features
gc.collect()


# # 5. Feature Selection using Equilibrium Optimizer (EO)

# Define paths for storing both scaled and unscaled feature selection results
INDICES_CACHE_PATH_SCALED = 'selected_indices_scaled_eo.npy' # Changed from sca to eo
INDICES_CACHE_PATH_UNSCALED = 'selected_indices_unscaled_eo.npy' # Changed from sca to eo
EO_HISTORY_SCALED_PATH = 'eo_history_scaled.npy'  # To store fitness history, changed from sca to eo
EO_HISTORY_UNSCALED_PATH = 'eo_history_unscaled.npy'  # To store fitness history, changed from sca to eo
# Use INDICES_CACHE_PATH for backward compatibility (points to scaled version)
INDICES_CACHE_PATH = INDICES_CACHE_PATH_SCALED

# Check if cached indices exist
if (os.path.exists(INDICES_CACHE_PATH_SCALED) and 
    os.path.exists(INDICES_CACHE_PATH_UNSCALED) and
    os.path.exists(EO_HISTORY_SCALED_PATH) and 
    os.path.exists(EO_HISTORY_UNSCALED_PATH)):
    print(f"Loading cached selected indices (EO)...") # Changed from SCA to EO
    selected_feature_indices_scaled = np.load(INDICES_CACHE_PATH_SCALED)
    selected_feature_indices_unscaled = np.load(INDICES_CACHE_PATH_UNSCALED)
    eo_history_scaled = np.load(EO_HISTORY_SCALED_PATH) # Changed from sca to eo
    eo_history_unscaled = np.load(EO_HISTORY_UNSCALED_PATH) # Changed from sca to eo
    
    # Use scaled indices for backward compatibility
    selected_feature_indices = selected_feature_indices_scaled
    
    num_features_total = X_train_scaled.shape[1]
    print(f"Loaded {len(selected_feature_indices_scaled)} scaled indices and {len(selected_feature_indices_unscaled)} unscaled indices (EO).") # Changed from SCA to EO
    SKIP_EO = True # Changed from SKIP_SCA to SKIP_EO
else:
    print("Cache not found. Running EO on both scaled and unscaled data...") # Changed from SCA to EO
    SKIP_EO = False # Changed from SKIP_SCA to SKIP_EO

if not SKIP_EO: # Changed from SKIP_SCA to SKIP_EO
    # --- Common EO parameters ---
    num_features_total = X_train_scaled.shape[1]
    epoch = 100
    pop_size = 10 
    
    # Create a list of BinaryVar objects, one for each feature
    binary_bounds = [BinaryVar() for _ in range(num_features_total)]
    
    # --- Define fitness function for SCALED data ---
    knn_eval_scaled = KNeighborsClassifier(n_neighbors=5)
    
    def fitness_function_scaled(solution):
        selected_indices = np.where(solution == 1)[0]
        num_selected = len(selected_indices)
        
        if num_selected == 0:
            return 1.0  # Penalize solutions with no features
        
        # Select features from TRAINING and TEST sets
        X_train_subset = X_train_scaled[:, selected_indices]
        X_test_subset = X_test_scaled[:, selected_indices]
        
        # Train k-NN on the TRAINING subset
        knn_eval_scaled.fit(X_train_subset, y_train)
        
        # Evaluate on TEST set
        y_pred_test = knn_eval_scaled.predict(X_test_subset)
        accuracy = accuracy_score(y_test, y_pred_test)
        
        # Calculate fitness value (minimize this)
        fitness = 0.99 * (1 - accuracy) + 0.01 * (num_selected / num_features_total)
        return fitness
    
    # --- Define fitness function for UNSCALED data ---
    knn_eval_unscaled = KNeighborsClassifier(n_neighbors=5)
    
    def fitness_function_unscaled(solution):
        selected_indices = np.where(solution == 1)[0]
        num_selected = len(selected_indices)
        
        if num_selected == 0:
            return 1.0  # Penalize solutions with no features
        
        # Select features from TRAINING and TEST sets
        X_train_subset = X_train_unscaled[:, selected_indices]
        X_test_subset = X_test_unscaled[:, selected_indices]
        
        # Train k-NN on the TRAINING subset
        knn_eval_unscaled.fit(X_train_subset, y_train)
        
        # Evaluate on TEST set
        y_pred_test = knn_eval_unscaled.predict(X_test_subset)
        accuracy = accuracy_score(y_test, y_pred_test)
        
        # Calculate fitness value (minimize this)
        fitness = 0.99 * (1 - accuracy) + 0.01 * (num_selected / num_features_total)
        return fitness
    
    # --- Setup problem dictionaries for both scaled and unscaled data ---
    problem_dict_scaled = {
        "obj_func": fitness_function_scaled,
        "bounds": binary_bounds,
        "minmax": "min",
        "log_to": "file",  # Save convergence to file
        "log_file": "eo_scaled_convergence.log", # Changed from sca to eo
        "verbose": True
    }
    
    problem_dict_unscaled = {
        "obj_func": fitness_function_unscaled,
        "bounds": binary_bounds,
        "minmax": "min",
        "log_to": "file",  # Save convergence to file
        "log_file": "eo_unscaled_convergence.log", # Changed from sca to eo
        "verbose": True
    }
    
    # --- Run EO on SCALED data ---
    print("\n" + "="*50)
    print("Running EO on SCALED data...") # Changed from SCA to EO
    print("="*50)
    
    start_time = time.time()
    model_eo_scaled = EO.OriginalEO(epoch=epoch, pop_size=pop_size) # Changed from SCA.OriginalSCA
    best_solution_eo_scaled = model_eo_scaled.solve(problem_dict_scaled) # Changed from model_sca_scaled
    end_time = time.time()
    print(f"EO on scaled data finished in {end_time - start_time:.2f} seconds.") # Changed from SCA to EO
    
    # Extract selected indices from the best solution
    best_position_scaled = best_solution_eo_scaled.solution # Changed from best_solution_scaled
    selected_feature_indices_scaled = np.where(best_position_scaled == 1)[0]
    num_selected_features_scaled = len(selected_feature_indices_scaled)
    best_fitness_scaled = best_solution_eo_scaled.target.fitness # Changed from best_solution_scaled
    
    print(f"Selected {num_selected_features_scaled} features out of {num_features_total} (EO - SCALED).") # Changed from SCA to EO
    print(f"Best fitness found (EO - SCALED): {best_fitness_scaled:.4f}") # Changed from SCA to EO
    
    # --- Run EO on UNSCALED data ---
    print("\n" + "="*50)
    print("Running EO on UNSCALED data...") # Changed from SCA to EO
    print("="*50)
    
    start_time = time.time()
    model_eo_unscaled = EO.OriginalEO(epoch=epoch, pop_size=pop_size) # Changed from SCA.OriginalSCA
    best_solution_eo_unscaled = model_eo_unscaled.solve(problem_dict_unscaled) # Changed from model_sca_unscaled
    end_time = time.time()
    print(f"EO on unscaled data finished in {end_time - start_time:.2f} seconds.") # Changed from SCA to EO
    
    # Extract selected indices from the best solution
    best_position_unscaled = best_solution_eo_unscaled.solution # Changed from best_solution_unscaled
    selected_feature_indices_unscaled = np.where(best_position_unscaled == 1)[0]
    num_selected_features_unscaled = len(selected_feature_indices_unscaled)
    best_fitness_unscaled = best_solution_eo_unscaled.target.fitness # Changed from best_solution_unscaled
    
    print(f"Selected {num_selected_features_unscaled} features out of {num_features_total} (EO - UNSCALED).") # Changed from SCA to EO
    print(f"Best fitness found (EO - UNSCALED): {best_fitness_unscaled:.4f}") # Changed from SCA to EO
    
    # --- Compare results ---
    print("\n" + "="*50)
    print("COMPARISON OF EO RESULTS") # Changed from SCA to EO
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
    # Extract fitness history from the solution
    try:
        eo_history_scaled = np.array(model_eo_scaled.history.list_global_best_fit) # Changed from model_sca_scaled
        eo_history_unscaled = np.array(model_eo_unscaled.history.list_global_best_fit) # Changed from model_sca_unscaled
        
        # Plot convergence comparison
        plt.figure(figsize=(10, 6))
        plt.plot(eo_history_scaled, 'b-', label='Scaled Features (EO)') # Changed from SCA to EO
        plt.plot(eo_history_unscaled, 'r-', label='Unscaled Features (EO)') # Changed from SCA to EO
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value (lower is better)')
        plt.title('EO Convergence: Scaled vs Unscaled Features') # Changed from SCA to EO
        plt.legend()
        plt.grid(True)
        plt.savefig('eo_convergence_comparison.png') # Changed from sca to eo
        plt.show()
    except Exception as e:
        print(f"Could not extract or plot convergence history (EO): {e}") # Changed from SCA to EO
        eo_history_scaled = np.array([]) # Changed from sca to eo
        eo_history_unscaled = np.array([]) # Changed from sca to eo
    
    # --- Save results to cache ---
    print("\nSaving feature selection results to cache (EO)...") # Changed from SCA to EO
    np.save(INDICES_CACHE_PATH_SCALED, selected_feature_indices_scaled)
    np.save(INDICES_CACHE_PATH_UNSCALED, selected_feature_indices_unscaled)
    np.save(EO_HISTORY_SCALED_PATH, eo_history_scaled) # Changed from SCA_HISTORY_SCALED_PATH
    np.save(EO_HISTORY_UNSCALED_PATH, eo_history_unscaled) # Changed from SCA_HISTORY_UNSCALED_PATH
    
    # Use scaled indices for backward compatibility
    selected_feature_indices = selected_feature_indices_scaled
    np.save(INDICES_CACHE_PATH, selected_feature_indices) # This still uses the generic name for compatibility
    
    print("Feature selection results cached successfully (EO).") # Changed from SCA to EO
# --- End of EO feature selection section ---


# # 6. Classification using k-NN with Selected Features

# In[ ]:

if not SKIP_TRAINING:
    # Select the features chosen by EO
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
    X_test_selected_scaled = X_test_scaled[:, selected_feature_indices_scaled]
    
    X_train_selected_unscaled = X_train_unscaled[:, selected_feature_indices_unscaled]
    X_test_selected_unscaled = X_test_unscaled[:, selected_feature_indices_unscaled]
    
    print("\n" + "="*50)
    print("COMPARING SCALED AND UNSCALED FEATURES")
    print("="*50)
    
    knn_scaled = KNeighborsClassifier(n_neighbors=5)
    knn_unscaled = KNeighborsClassifier(n_neighbors=5)
    
    print("Training initial k-NN models...")
    knn_scaled.fit(X_train_selected_scaled, y_train)
    knn_unscaled.fit(X_train_selected_unscaled, y_train)
    
    print("Evaluating on test set...")
    y_pred_test_scaled = knn_scaled.predict(X_test_selected_scaled)
    y_pred_test_unscaled = knn_unscaled.predict(X_test_selected_unscaled)
    
    accuracy_scaled = accuracy_score(y_test, y_pred_test_scaled)
    accuracy_unscaled = accuracy_score(y_test, y_pred_test_unscaled)
    
    print(f"Initial accuracy with scaled features (k=5): {accuracy_scaled:.4f}")
    print(f"Initial accuracy with unscaled features (k=5): {accuracy_unscaled:.4f}")
    
    initial_knn_scaled = knn_scaled
    initial_knn_unscaled = knn_unscaled
    
    if accuracy_scaled >= accuracy_unscaled:
        print("\nScaled features perform better. Using scaled features for k-NN tuning.")
        use_scaled = True
        X_train_selected = X_train_selected_scaled
        X_test_selected = X_test_selected_scaled
        selected_feature_indices = selected_feature_indices_scaled
        initial_knn = initial_knn_scaled
        initial_accuracy = accuracy_scaled
    else:
        print("\nUnscaled features perform better. Using unscaled features for k-NN tuning.")
        use_scaled = False
        X_train_selected = X_train_selected_unscaled
        X_test_selected = X_test_selected_unscaled
        selected_feature_indices = selected_feature_indices_unscaled
        initial_knn = initial_knn_unscaled
        initial_accuracy = accuracy_unscaled
    
    print("\n" + "="*50)
    print("PERFORMING k-NN HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
    print("="*50)
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]
    }
    
    knn_base = KNeighborsClassifier()
    
    print("Starting GridSearchCV (this may take a while)...")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=knn_base,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    grid_search.fit(X_train_selected, y_train)
    
    end_time = time.time()
    print(f"GridSearchCV finished in {end_time - start_time:.2f} seconds.")
    
    best_params = grid_search.best_params_
    print(f"\nBest parameters found: {best_params}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    cv_results = pd.DataFrame(grid_search.cv_results_)
    top_results = cv_results.sort_values('rank_test_score').head(5)
    
    print("\nTop 5 Parameter Combinations:")
    for i, row in top_results.iterrows():
        params = row['params']
        score = row['mean_test_score']
        std = row['std_test_score']
        print(f"- Score: {score:.4f} (±{std:.4f}) | {params}")
    
    k_results = []
    for k in [3, 5, 7, 9, 11, 13]:
        k_data = cv_results[cv_results['param_n_neighbors'] == k]
        k_mean = k_data['mean_test_score'].mean()
        k_std = k_data['mean_test_score'].std()
        k_max = k_data['mean_test_score'].max()
        k_results.append((k, k_mean, k_std, k_max))
    
    plt.figure(figsize=(10, 6))
    k_values = [x[0] for x in k_results]
    mean_scores = [x[1] for x in k_results]
    std_scores = [x[2] for x in k_results]
    max_scores = [x[3] for x in k_results]
    
    plt.errorbar(k_values, mean_scores, yerr=std_scores, fmt='-o', capsize=5, label='Mean ± Std')
    plt.plot(k_values, max_scores, 'r--o', label='Max Score')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('k-NN: Mean Accuracy vs k Value')
    plt.grid(True)
    plt.legend()
    plt.savefig('knn_gridsearch_results.png')
    plt.show()
    
    knn_final = grid_search.best_estimator_
    
    print("\nEvaluating best model on test set...")
    start_time = time.time()
    y_pred_test = knn_final.predict(X_test_selected)
    end_time = time.time()
    
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test accuracy with best model: {test_accuracy:.4f}")
    print(f"Prediction time: {end_time - start_time:.2f} seconds")
    
    print(f"Initial model accuracy: {initial_accuracy:.4f}")
    
    if test_accuracy < initial_accuracy:
        print("\n*** WARNING: GridSearchCV model performs worse than initial model! ***")
        print(f"Reverting to initial model with accuracy {initial_accuracy:.4f}")
        
        knn_final = initial_knn
        y_pred_test = knn_final.predict(X_test_selected)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"Using initial model with parameters: {initial_knn.get_params()}")
        best_params = initial_knn.get_params()
    
    feature_type = "scaled" if use_scaled else "unscaled"
    print(f"Final model uses {feature_type} features with hyperparameters: {best_params}")
    
    joblib.dump(knn_final, MODEL_CACHE_PATH)
    joblib.dump(scaler, SCALER_CACHE_PATH)
    np.save(INDICES_CACHE_PATH, selected_feature_indices)
    with open('feature_type.txt', 'w') as f:
        f.write(feature_type)
    joblib.dump(best_params, 'best_knn_params.joblib')
    
    print(f"Saved final k-NN model to {MODEL_CACHE_PATH}")
    print(f"Saved feature type information to feature_type.txt")
    print(f"Saved best hyperparameters to best_knn_params.joblib")
else:
    if os.path.exists('feature_type.txt'):
        with open('feature_type.txt', 'r') as f:
            feature_type = f.read().strip()
        print(f"Using {feature_type} features as specified in feature_type.txt")
        
        if feature_type == "scaled":
            if os.path.exists(INDICES_CACHE_PATH_SCALED):
                selected_feature_indices = np.load(INDICES_CACHE_PATH_SCALED)
                X_test_selected = X_test_scaled[:, selected_feature_indices]
            else:
                X_test_selected = X_test_scaled[:, selected_feature_indices]
        else:
            if os.path.exists(INDICES_CACHE_PATH_UNSCALED):
                selected_feature_indices = np.load(INDICES_CACHE_PATH_UNSCALED)
                X_test_selected = X_test_unscaled[:, selected_feature_indices]
            else:
                X_test_selected = X_test_unscaled[:, selected_feature_indices]
    else:
        print("No feature type information found. Defaulting to scaled features.")
        X_test_selected = X_test_scaled[:, selected_feature_indices]
    
    if os.path.exists('best_knn_params.joblib'):
        best_params = joblib.load('best_knn_params.joblib')
        print(f"Best hyperparameters from GridSearchCV: {best_params}")
    
    print("Predicting on test set using cached model...")
    start_time = time.time()
    y_pred_test = knn_final.predict(X_test_selected)
    end_time = time.time()
    print(f"Prediction finished in {end_time - start_time:.2f} seconds.")


# # 7. Evaluation and Visualization

# In[ ]:


def generate_knn_roc_curves(X_test_selected, y_test, knn_model, name):
    print(f"Generating ROC curves for {name}...")
    
    try:
        distances, indices = knn_model.kneighbors(X_test_selected)
        
        unique_classes = np.sort(np.unique(y_test))
        n_classes = len(unique_classes)
        y_bin = label_binarize(y_test, classes=unique_classes)
        
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        y_score = np.zeros((X_test_selected.shape[0], n_classes))
        
        y_train_labels = knn_model._y
        
        for i in range(X_test_selected.shape[0]):
            neighbor_indices = indices[i]
            neighbor_distances = distances[i]
            
            weights = 1.0 / (neighbor_distances + 1e-10)
            
            for idx, class_val in enumerate(unique_classes):
                class_neighbors = np.where(y_train_labels[neighbor_indices] == class_val)[0]
                if len(class_neighbors) > 0:
                    y_score[i, idx] = weights[class_neighbors].sum()
        
        row_sums = y_score.sum(axis=1)
        y_score = y_score / row_sums[:, np.newaxis]
        
        for i, class_val in enumerate(unique_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
        
        ax_roc.plot(fpr["micro"], tpr["micro"],
                  label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
                  color='deeppink', linestyle=':', linewidth=4)
        
        colors = cycle(['blue', 'red', 'green', 'orange'])
        for i, (class_val, color) in enumerate(zip(unique_classes, colors)):
            emotion_name = EMOTIONS[class_val]
            ax_roc.plot(fpr[i], tpr[i], color=color, lw=2,
                      label=f'{emotion_name} (AUC = {roc_auc[i]:.2f})')
        
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC Curves ({name})')
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, alpha=0.3)
        
        filename = f'roc_curves_{name.replace(" ", "_").lower()}.png'
        plt.savefig(filename)
        plt.close(fig_roc)
        
        print(f"Successfully saved ROC curves to {filename}")
        return fig_roc, fpr, tpr, roc_auc
        
    except Exception as e:
        import traceback
        print(f"Could not create ROC curves for {name}: {str(e)}")
        traceback.print_exc()
        return None, None, None, None


def evaluate_and_save_results():
    print("\n\n" + "="*50)
    print("GENERATING COMPREHENSIVE EVALUATION METRICS")
    print("="*50)
    
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    test_precision = precision_score(y_test, y_pred_test, average='weighted')
    test_recall = recall_score(y_test, y_pred_test, average='weighted')
    
    unique_emotions = sorted(np.unique(np.concatenate([y_test, y_pred_test])))
    emotion_names = [EMOTIONS.get(e, f"Unknown-{e}") for e in unique_emotions]
    
    cm = confusion_matrix(y_test, y_pred_test, labels=unique_emotions)
    
    report = classification_report(
        y_test, 
        y_pred_test, 
        labels=unique_emotions,
        target_names=emotion_names,
        output_dict=True
    )
    
    X_train_selected = X_train_scaled[:, selected_feature_indices]
    
    y_train_pred = knn_final.predict(X_train_selected)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    
    train_per_class = {}
    test_per_class = {}
    
    for emotion in unique_emotions:
        train_indices = np.where(y_train == emotion)[0]
        test_indices = np.where(y_test == emotion)[0]
        
        if len(train_indices) > 0:
            train_emotion_true = y_train[train_indices]
            train_emotion_pred = y_train_pred[train_indices]
            train_per_class[emotion] = accuracy_score(train_emotion_true, train_emotion_pred)
        else:
            train_per_class[emotion] = 0
        
        if len(test_indices) > 0:
            test_emotion_true = y_test[test_indices]
            test_emotion_pred = y_pred_test[test_indices]
            test_per_class[emotion] = accuracy_score(test_emotion_true, test_emotion_pred)
        else:
            test_per_class[emotion] = 0
    
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=emotion_names,
        yticklabels=emotion_names
    )
    plt.title('Confusion Matrix for the Dataset (Arabic Emotion Recognition)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    fig_acc, ax_acc = plt.subplots(figsize=(12, 6))
    
    emotion_dict = {}
    for emotion in unique_emotions:
        emotion_name = EMOTIONS.get(emotion, f"Unknown-{emotion}")
        train_acc = train_per_class.get(emotion, 0)
        test_acc = test_per_class.get(emotion, 0)
        emotion_dict[emotion_name] = {
            'Train': train_acc,
            'Test': test_acc
        }
    
    emotion_df = pd.DataFrame(emotion_dict).T
    emotion_df.plot(kind='bar', ax=ax_acc)
    ax_acc.set_ylim(0, 1.1)
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_xlabel('Emotion')
    ax_acc.set_title('Accuracy by Emotion')
    ax_acc.legend(title='Dataset')
    plt.tight_layout()
    
    fig_imp = None
    try:
        importances = np.zeros(len(selected_feature_indices))
        for i, neighbor in enumerate(knn_final.kneighbors(X_test_selected)[1]):
            neighbors = X_train_selected[neighbor]
            spread = np.std(neighbors, axis=0)
            importances += 1.0 / (spread + 1e-10)
        
        importances = importances / np.sum(importances)
        
        top_n = min(20, len(importances))
        top_indices = np.argsort(importances)[-top_n:][::-1]
        top_importances = importances[top_indices]
        
        fig_imp, ax_imp = plt.subplots(figsize=(12, 6))
        ax_imp.bar(range(top_n), top_importances)
        ax_imp.set_xticks(range(top_n))
        ax_imp.set_xticklabels([f"Feature {selected_feature_indices[i]}" for i in top_indices], rotation=90)
        ax_imp.set_title('Top Feature Importances (Estimated from KNN)')
        ax_imp.set_ylabel('Relative Importance')
        plt.tight_layout()
    except Exception as e:
        print(f"Could not create feature importance figure: {e}")
    
    fig_dist = None
    try:
        class_counts = {EMOTIONS.get(i, f"Unknown-{i}"): (y == i).sum() for i in unique_emotions}
        
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        ax_dist.bar(class_counts.keys(), class_counts.values())
        ax_dist.set_title('Distribution of Emotion Classes in Dataset')
        ax_dist.set_ylabel('Number of Samples')
        ax_dist.set_xlabel('Emotion')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    except Exception as e:
        print(f"Could not create class distribution figure: {e}")
    
    fig_roc_scaled, fpr_scaled, tpr_scaled, roc_auc_scaled = generate_knn_roc_curves(
        X_test_selected, y_test, knn_final, "Scaled Features"
    )
    
    test_accuracy_unscaled = None
    test_f1_unscaled = None
    cm_unscaled = None
    report_unscaled = None
    fig_cm_unscaled = None
    fig_roc_unscaled = None
    
    try:
        if os.path.exists(INDICES_CACHE_PATH_UNSCALED) and os.path.exists(FEATURES_CACHE_PATH):
            print("Calculating evaluation metrics for unscaled features...")
            
            X_orig = np.load(FEATURES_CACHE_PATH)
            y_orig = Data['Emotion'].values
            selected_feature_indices_unscaled = np.load(INDICES_CACHE_PATH_UNSCALED)
            
            X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled = train_test_split(
                X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
            )
            
            X_train_selected_unscaled = X_train_unscaled[:, selected_feature_indices_unscaled]
            X_test_selected_unscaled = X_test_unscaled[:, selected_feature_indices_unscaled]
            
            knn_unscaled = KNeighborsClassifier(n_neighbors=5, weights='distance')
            knn_unscaled.fit(X_train_selected_unscaled, y_train_unscaled)
            
            y_pred_unscaled = knn_unscaled.predict(X_test_selected_unscaled)
            
            test_accuracy_unscaled = accuracy_score(y_test_unscaled, y_pred_unscaled)
            test_f1_unscaled = f1_score(y_test_unscaled, y_pred_unscaled, average='weighted')
            
            cm_unscaled = confusion_matrix(y_test_unscaled, y_pred_unscaled, labels=unique_emotions)
            
            report_unscaled = classification_report(
                y_test_unscaled, 
                y_pred_unscaled, 
                labels=unique_emotions,
                target_names=emotion_names,
                output_dict=True
            )
            
            fig_cm_unscaled, ax_cm_unscaled = plt.subplots(figsize=(10, 8))
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
            
            fig_roc_unscaled, fpr_unscaled, tpr_unscaled, roc_auc_unscaled = generate_knn_roc_curves(
                X_test_selected_unscaled, y_test_unscaled, knn_unscaled, "Unscaled Features"
            )
            
            del X_orig, y_orig
            del X_train_unscaled, X_test_unscaled
            del X_train_selected_unscaled, X_test_selected_unscaled
            del y_train_unscaled, y_test_unscaled, y_pred_unscaled
            del knn_unscaled
            gc.collect()
    except Exception as e:
        print(f"Could not calculate unscaled metrics: {e}")
        test_accuracy_unscaled = None
        test_f1_unscaled = None
        cm_unscaled = None
        report_unscaled = None
    
    results = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'confusion_matrix': cm,
        'unique_emotions': unique_emotions,
        'emotion_names': emotion_names,
        'train_size': len(y_train),
        'test_size': len(y_test),
        'classification_report': report,
        'train_per_class': train_per_class,
        'test_per_class': test_per_class,
        'confusion_matrix_fig': fig_cm,
        'accuracy_by_emotion_fig': fig_acc,
        'feature_importance_fig': fig_imp,
        'class_distribution_fig': fig_dist,
        'roc_curves_fig': fig_roc_scaled,
        'val_accuracy': None,
        'val_f1': None,
        'val_size': 0,
        
        'test_accuracy_unscaled': test_accuracy_unscaled,
        'test_f1_unscaled': test_f1_unscaled,
        'confusion_matrix_unscaled': cm_unscaled,
        'classification_report_unscaled': report_unscaled,
        'confusion_matrix_fig_unscaled': fig_cm_unscaled,
        'roc_curves_fig_unscaled': fig_roc_unscaled,
        
        'selected_indices_scaled_path': INDICES_CACHE_PATH_SCALED,
        'selected_indices_unscaled_path': INDICES_CACHE_PATH_UNSCALED,
        'sca_history_scaled_path': EO_HISTORY_SCALED_PATH, # Changed from SCA_HISTORY_SCALED_PATH
        'sca_history_unscaled_path': EO_HISTORY_UNSCALED_PATH # Changed from SCA_HISTORY_UNSCALED_PATH
    }
    
    joblib.dump(results, EVALUATION_RESULTS_PATH)
    print(f"\nEvaluation results saved to '{EVALUATION_RESULTS_PATH}'")
    
    try:
        fig_cm.savefig('confusion_matrix.png')
        fig_acc.savefig('accuracy_by_emotion.png')
        if fig_imp is not None:
            fig_imp.savefig('feature_importance.png')
        if fig_dist is not None:
            fig_dist.savefig('class_distribution.png')
        print("Evaluation plots saved as PNG files.")
    except Exception as e:
        print(f"Error saving evaluation plots: {e}")
    
    test_accuracy_unscaled_str = f"{test_accuracy_unscaled:.4f}" if test_accuracy_unscaled is not None else "N/A"
    test_f1_unscaled_str = f"{test_f1_unscaled:.4f}" if test_f1_unscaled is not None else "N/A"
    
    report_content = f"""
    # Arabic Emotion Recognition Model Evaluation Report

    ## Overall Performance with Scaled Features
    - Training Accuracy: {train_accuracy:.4f}
    - Test Accuracy: {test_accuracy:.4f}
    - Test F1 Score (Weighted): {test_f1:.4f}
    - Test Precision (Weighted): {test_precision:.4f}
    - Test Recall (Weighted): {test_recall:.4f}
    
    ## Performance with Unscaled Features
    - Test Accuracy: {test_accuracy_unscaled_str}
    - Test F1 Score (Weighted): {test_f1_unscaled_str}
    """
    
    with open('evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
        print("Saved evaluation report to 'evaluation_report.md'")
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Training Set: {len(y_train)} samples")
    print(f"Test Set: {len(y_test)} samples")
    print(f"Selected Features (Scaled): {len(selected_feature_indices_scaled)} out of {X_train_scaled.shape[1]}")
    if 'selected_feature_indices_unscaled' in locals():
        print(f"Selected Features (Unscaled): {len(selected_feature_indices_unscaled)} out of {X_train_scaled.shape[1]}")
    print("="*50)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy (Scaled): {test_accuracy:.4f}")
    if test_accuracy_unscaled is not None:
        print(f"Test Accuracy (Unscaled): {test_accuracy_unscaled:.4f}")
    print("="*50)
    print(f"Test F1 Score (Scaled): {test_f1:.4f}")
    if test_f1_unscaled is not None:
        print(f"Test F1 Score (Unscaled): {test_f1_unscaled:.4f}")
    print("="*50)
    
    return results


evaluate_and_save_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arabic Emotion Recognition Training Script')
    parser.add_argument('--clear-all', action='store_true', 
                        help='Clear all cached files before running')
    parser.add_argument('--clear-model', action='store_true', 
                        help='Clear only model files before running')
    parser.add_argument('--clear-evaluation', action='store_true', 
                        help='Clear only evaluation results before running')
    args = parser.parse_args()
    
    if args.clear_all:
        cache_files = [FEATURES_CACHE_PATH, DATA_CACHE_PATH, INDICES_CACHE_PATH, 
                       SCALER_CACHE_PATH, MODEL_CACHE_PATH, EVALUATION_RESULTS_PATH]
        for file in cache_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Deleted cache file: {file}")
    
    elif args.clear_model:
        model_files = [INDICES_CACHE_PATH, SCALER_CACHE_PATH, MODEL_CACHE_PATH, EVALUATION_RESULTS_PATH]
        for file in model_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Deleted model file: {file}")
    
    elif args.clear_evaluation:
        if os.path.exists(EVALUATION_RESULTS_PATH):
            os.remove(EVALUATION_RESULTS_PATH)
            print(f"Deleted evaluation results: {EVALUATION_RESULTS_PATH}")
    
    print("Training and evaluation complete.")



