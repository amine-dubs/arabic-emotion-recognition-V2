#!/usr/bin/env python
# coding: utf-8

# Arabic Emotion Recognition using Parallel CNN with EO-optimized Convolutional Layers
# Based on the Parallel CNN Transformer with addition of Equilibrium Optimizer for CNN parameters

import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time
import joblib
import gc
import argparse
import sys
import traceback

print("Script started")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

# Imports for audio processing and visualization
import tensorflow as tf

# Import EO optimization algorithm
from mealpy import EO
from mealpy.utils.space import BinaryVar

# Import for model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# PyTorch imports for CNN model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Cache File Paths ---
FEATURES_CACHE_PATH = 'mel_spectrograms.npy'
DATA_CACHE_PATH = 'data_df.pkl'
INDICES_CACHE_PATH_EO = 'cnn_params_eo.npy'
MODEL_CACHE_PATH = 'cnn_eo_final_model.pt'
EVALUATION_RESULTS_PATH = 'evaluation_results_cnn_eo.joblib'
EO_HISTORY_PATH = 'cnn_eo_history.npy'
# ------------------------

EMOTIONS = {0: 'Angry',
           2: 'Happy',
           3: 'Neutral',
           4: 'Sad'}

# Map the original emotion indices to consecutive indices (0, 1, 2, 3)
EMOTION_MAPPING = {0: 0, 2: 1, 3: 2, 4: 3}
REVERSE_MAPPING = {0: 0, 1: 2, 2: 3, 3: 4}

# Map emotion names to consecutive indices
EMOTION_NAMES = ['Angry', 'Happy', 'Neutral', 'Sad']

DATA_PATH = './Data'
SAMPLE_RATE = 16000
DURATION = 3  # seconds

# --- Check for cached features and DataFrame ---
SKIP_FEATURE_EXTRACTION = False
if os.path.exists(FEATURES_CACHE_PATH) and os.path.exists(DATA_CACHE_PATH):
    print(f"Checking cached features from {FEATURES_CACHE_PATH}...")
    try:
        # Try to load the cached features
        mel_spectrograms = np.load(FEATURES_CACHE_PATH, allow_pickle=True)
        
        # Verify that the loaded data is valid
        if mel_spectrograms is not None and mel_spectrograms.size > 0:
            print(f"Loading cached DataFrame from {DATA_CACHE_PATH}...")
            Data = pd.read_pickle(DATA_CACHE_PATH)
            print("Loaded features shape:", mel_spectrograms.shape)
            print("Loaded DataFrame head:\n", Data.head())
            
            # Check if shapes are compatible
            if len(mel_spectrograms) == len(Data):
                # Skip feature extraction
                SKIP_FEATURE_EXTRACTION = True
            else:
                print(f"WARNING: Cached features length ({len(mel_spectrograms)}) doesn't match DataFrame length ({len(Data)})")
                print("Will regenerate features...")
        else:
            print("WARNING: Cached features array is empty. Will regenerate features...")
    except Exception as e:
        print(f"ERROR loading cached data: {e}")
        print("Will regenerate features...")

if not SKIP_FEATURE_EXTRACTION:
    print("Starting data loading and feature extraction...")
# ---------------------------------------------

# --- Check for cached model ---
if os.path.exists(MODEL_CACHE_PATH):
    print(f"Loading cached model from {MODEL_CACHE_PATH}...")
    # Will load model later
    SKIP_TRAINING = True
else:
    print("No cached model found. Will train a new model.")
    SKIP_TRAINING = False
# ---------------------------------------------

# Define MEL spectrogram extraction function
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
def addAWGN(signal, num_bits=16, augmented_num=2, snr_low=15, snr_high=30): 
    signal_len = len(signal)
    # Generate White Gaussian noise
    noise = np.random.normal(size=(augmented_num, signal_len))
    # Normalize signal and noise
    norm_constant = 2.0**(num_bits-1)
    signal_norm = signal / norm_constant
    noise_norm = noise / norm_constant
    # Compute signal and noise power
    s_power = np.sum(signal_norm ** 2) / signal_len
    n_power = np.sum(noise_norm ** 2, axis=1) / signal_len
    # Random SNR: Uniform [15, 30] in dB
    target_snr = np.random.randint(snr_low, snr_high)
    # Compute K (covariance matrix) for each noise 
    K = np.sqrt((s_power / n_power) * 10 ** (- target_snr / 10))
    K = np.ones((signal_len, augmented_num)) * K  
    # Generate noisy signal
    return signal + K.T * noise

# --- Skip spectrogram generation if features are loaded ---
if not SKIP_FEATURE_EXTRACTION:
    file_names = []
    file_emotions = []
    file_paths = []
    
    print(f"DATA_PATH = {DATA_PATH}")
    print(f"Checking if DATA_PATH exists: {os.path.exists(DATA_PATH)}")
    
    # Debug: list top-level directories
    print("Top-level directories in DATA_PATH:")
    if os.path.exists(DATA_PATH):
        for item in os.listdir(DATA_PATH):
            print(f"  - {item}")

    # Iterate over each show folder (e.g., EYASE)
    for show_folder in os.listdir(DATA_PATH):
        show_path = os.path.join(DATA_PATH, show_folder)
        if not os.path.isdir(show_path):
            continue
        
        # Iterate over actor folders
        for actor_folder in os.listdir(show_path):
            actor_path = os.path.join(show_path, actor_folder)
            if not os.path.isdir(actor_path):
                continue
            
            # Process all WAV files in the actor folder
            print(f"Processing files in {actor_path}")
            for file_name in os.listdir(actor_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(actor_path, file_name)
                    
                    # Extract emotion from filename (assuming format like "fm01_ang (1).wav")
                    emotion_code = None
                    
                    if '_ang ' in file_name.lower():
                        emotion_code = 0  # Angry
                    elif '_hap ' in file_name.lower():
                        emotion_code = 2  # Happy
                    elif '_neu ' in file_name.lower():
                        emotion_code = 3  # Neutral
                    elif '_sad ' in file_name.lower():
                        emotion_code = 4  # Sad
                    
                    if emotion_code is not None:
                        file_names.append(file_name)
                        file_emotions.append(emotion_code)
                        file_paths.append(file_path)
                    
            # Print count of collected files after each actor folder
            print(f"Total files collected so far: {len(file_paths)}")
                        
    # Create a DataFrame
    Data = pd.DataFrame({
        "Name": file_names,
        "Emotion": file_emotions,
        "Path": file_paths
    })
    # --- Save DataFrame to cache ---
    print(f"Saving DataFrame to {DATA_CACHE_PATH}...")
    Data.to_pickle(DATA_CACHE_PATH)

    print("Number of files loaded:", len(Data))    # Generate MEL spectrograms
    mel_spectrograms = []
    signals = []
    error_files = []
    processed_files = []
    
    start_time = time.time()
    print("Generating MEL spectrograms...")
    print(f"Total files to process: {len(Data)}")
    
    # Process a small test batch first to verify it's working
    test_sample_size = min(3, len(Data))
    print(f"Testing spectrograms with {test_sample_size} files first...")
    
    for i in range(test_sample_size):
        file_path = Data.Path.iloc[i]
        print(f"Test processing file: {file_path}")
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"WARNING: Test file not found: {file_path}")
                continue
                
            print(f"Loading audio with librosa...")
            audio, sample_rate = librosa.load(file_path, duration=DURATION, offset=0.5, sr=SAMPLE_RATE)
            print(f"Audio loaded, shape: {audio.shape}, sample_rate: {sample_rate}")
            
            print(f"Creating signal array...")
            signal = np.zeros((int(SAMPLE_RATE*DURATION)))
            signal[:len(audio)] = audio
            
            print(f"Generating MEL spectrogram...")
            mel_spectrogram = getMELspectrogram(signal, sample_rate=SAMPLE_RATE)
            print(f"MEL spectrogram shape: {mel_spectrogram.shape}")
            
            # Test successful
            print(f"Test successful for file: {file_path}")
        except Exception as e:
            print(f"ERROR in test processing file {file_path}: {e}")
            traceback.print_exc()
    
    print("\nStarting full processing...")
    
    for i, file_path in enumerate(Data.Path):
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"\nWARNING: File not found: {file_path}")
                error_files.append(file_path)
                continue
            
            # Load audio file    
            audio, sample_rate = librosa.load(file_path, duration=DURATION, offset=0.5, sr=SAMPLE_RATE)
            
            # Create fixed-length signal array and fill with audio data
            signal = np.zeros((int(SAMPLE_RATE*DURATION)))
            signal[:len(audio)] = audio
            signals.append(signal)
            
            # Generate MEL spectrogram
            mel_spectrogram = getMELspectrogram(signal, sample_rate=SAMPLE_RATE)
            mel_spectrograms.append(mel_spectrogram)
            processed_files.append(file_path)
            
            # Print progress
            if i % 10 == 0 or i == len(Data) - 1:  # Update every 10 files or at the end
                print(f"\r Processed {i+1}/{len(Data)} files | Successful: {len(mel_spectrograms)}", end='')
        except Exception as e:
            print(f"\nERROR processing file {file_path}: {e}")
            error_files.append(file_path)
            continue
    
    # Verify we've processed at least one file
    if len(mel_spectrograms) == 0:
        print("\n\nDEBUG INFORMATION:")
        print(f"DATA_PATH: {DATA_PATH}")
        print(f"Data DataFrame shape: {Data.shape}")
        print(f"First few file paths: {Data.Path.iloc[:5].tolist()}")
        print(f"Error files count: {len(error_files)}")
        if error_files:
            print(f"First few error files: {error_files[:5]}")
            for path in error_files[:2]:
                print(f"File exists? {os.path.exists(path)}")
                if os.path.exists(path):
                    print(f"File size: {os.path.getsize(path)} bytes")
                    print(f"File content (hex): {open(path, 'rb').read(100).hex()}")
        print("\nCheck your audio file paths and librosa installation.")
            
    
    # Check if we have any spectrograms
    if len(mel_spectrograms) == 0:
        raise ValueError("No spectrograms were successfully extracted! Check your data path and files.")
    
    print(f"\nSuccessfully processed {len(mel_spectrograms)} files out of {len(Data)} total files.")
    if error_files:
        print(f"Failed to process {len(error_files)} files.")
    
    print("\nApplying data augmentation...")
    # Apply data augmentation
    original_count = len(signals)
    for i, signal in enumerate(signals[:original_count]):  # Only augment original signals
        try:
            augmented_signals = addAWGN(signal)
            for j in range(augmented_signals.shape[0]):
                mel_spectrogram = getMELspectrogram(augmented_signals[j,:], sample_rate=SAMPLE_RATE)
                mel_spectrograms.append(mel_spectrogram)
                Data = pd.concat([Data, Data.iloc[i:i+1]], ignore_index=True)
            print(f"\r Augmented {i+1}/{original_count} files", end='')
        except Exception as e:
            print(f"\nERROR during augmentation of signal {i}: {e}")
            continue
    
    # Check if we have enough data after augmentation
    if len(mel_spectrograms) == 0:
        raise ValueError("No spectrograms were successfully created after augmentation!")
        
    print(f"\nTotal spectrograms after augmentation: {len(mel_spectrograms)}")
    
    # Stack the spectrograms
    try:
        mel_spectrograms = np.stack(mel_spectrograms, axis=0)
        end_time = time.time()
        print(f"Finished extracting spectrograms in {end_time - start_time:.2f} seconds.")
        print("Shape of spectrograms array:", mel_spectrograms.shape)
    except Exception as e:
        print(f"ERROR stacking spectrograms: {e}")
        # Print debug information about the mel_spectrograms list
        print(f"Number of spectrograms: {len(mel_spectrograms)}")
        if len(mel_spectrograms) > 0:
            print(f"First spectrogram shape: {mel_spectrograms[0].shape}")
        raise
    
    # Save the extracted spectrograms
    np.save(FEATURES_CACHE_PATH, mel_spectrograms)
    print(f"Saved spectrograms to {FEATURES_CACHE_PATH}")
    
    # Free up memory
    del signals
    gc.collect()

# Make spectrograms 4D for CNN input: [batch, channels, height, width]
print("Checking mel_spectrograms shape:", mel_spectrograms.shape)

# Verify that mel_spectrograms is not empty
if mel_spectrograms.size == 0:
    raise ValueError("Error: mel_spectrograms is empty! Feature extraction failed.")

# Add channel dimension if necessary
if len(mel_spectrograms.shape) == 3:  # If shape is [batch, height, width]
    mel_spectrograms = np.expand_dims(mel_spectrograms, 1)  # Add channel dimension
    print("Added channel dimension, new shape:", mel_spectrograms.shape)

print("Spectrogram shape with channel:", mel_spectrograms.shape)

# Define the Parallel CNN model with flexible hyperparameters
class ParallelModel(nn.Module):
    def __init__(self, num_emotions, params):
        super(ParallelModel, self).__init__()
        # Extract hyperparameters from params dictionary
        self.conv1_filters = int(params.get('conv1_filters', 16))
        self.conv2_filters = int(params.get('conv2_filters', 32))
        self.conv3_filters = int(params.get('conv3_filters', 64))
        self.conv4_filters = int(params.get('conv4_filters', 64))
        self.kernel_size = int(params.get('kernel_size', 3))
        self.dropout_rate = params.get('dropout_rate', 0.3)
        self.transf_heads = int(params.get('transf_heads', 4))
        # Always use exactly 4 transformer layers as requested
        self.transf_layers = 4
        # Make sure transformer dimension is divisible by number of heads
        self.transf_dim_original = int(params.get('transf_dim', 64))
        # Adjust transformer dimension to be divisible by number of heads
        self.transf_dim = (self.transf_dim_original // self.transf_heads) * self.transf_heads
        # Ensure minimum size
        if self.transf_dim < self.transf_heads:
            self.transf_dim = self.transf_heads
        # Log the adjustment if it was made
        if self.transf_dim != self.transf_dim_original:
            print(f"Adjusted transformer dimension from {self.transf_dim_original} to {self.transf_dim} to be divisible by {self.transf_heads} heads")
        
        self.transf_ff_dim = int(params.get('transf_ff_dim', 512))
        
        # Conv block
        self.conv2Dblock = nn.Sequential(
            # 1st conv block
            nn.Conv2d(in_channels=1,
                     out_channels=self.conv1_filters,
                     kernel_size=self.kernel_size,
                     stride=1,
                     padding=1),
            nn.BatchNorm2d(self.conv1_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=self.dropout_rate),
            
            # 2nd conv block
            nn.Conv2d(in_channels=self.conv1_filters,
                     out_channels=self.conv2_filters,
                     kernel_size=self.kernel_size,
                     stride=1,
                     padding=1),
            nn.BatchNorm2d(self.conv2_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Changed from 4 to 2
            nn.Dropout(p=self.dropout_rate),
            
            # 3rd conv block
            nn.Conv2d(in_channels=self.conv2_filters,
                     out_channels=self.conv3_filters,
                     kernel_size=self.kernel_size,
                     stride=1,
                     padding=1),
            nn.BatchNorm2d(self.conv3_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Changed from 4 to 2
            nn.Dropout(p=self.dropout_rate),
            
            # 4th conv block
            nn.Conv2d(in_channels=self.conv3_filters,
                     out_channels=self.conv4_filters,
                     kernel_size=self.kernel_size,
                     stride=1,
                     padding=1),
            nn.BatchNorm2d(self.conv4_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Changed from 4 to 2
            nn.Dropout(p=self.dropout_rate)
        )          # Transformer block
        self.transf_maxpool = nn.MaxPool2d(kernel_size=[2,2], stride=[2,2])  # Reduced stride
        transf_layer = nn.TransformerEncoderLayer(
            d_model=self.transf_dim, 
            nhead=self.transf_heads, 
            dim_feedforward=256,  # Reduced from self.transf_ff_dim to fixed 256 for efficiency
            dropout=self.dropout_rate, 
            activation='relu'
        )
        self.transf_encoder = nn.TransformerEncoder(transf_layer, num_layers=self.transf_layers)
        
        # Calculate the flattened dimension of the CNN output
        self.cnn_flatten_dim = self._calculate_cnn_output_dim()
        print(f"Estimated CNN flattened dimension: {self.cnn_flatten_dim}")
        
        # Linear layer dimensions - either use calculated dimension or a default safe value
        cnn_dim = self.cnn_flatten_dim if self.cnn_flatten_dim > 0 else 256
        
        # Linear softmax layer
        # Size of output from conv block + transformer output
        self.out_linear = nn.Linear(cnn_dim + self.transf_dim, num_emotions)
        self.dropout_linear = nn.Dropout(p=self.dropout_rate)
        self.out_softmax = nn.Softmax(dim=1)
        
        # Initialize flag for shape logging
        self.shapes_logged = False
        
    def _calculate_cnn_output_dim(self):
        """
        Estimate the output dimension of the CNN after flattening.
        This is an approximation and may need adjustment based on actual input dimensions.
        """
        try:
            # Assuming input shape of [batch, channels, height, width] = [1, 1, 128, 188]
            # These are the dimensions from our mel spectrogram
            h, w = 128, 188  # Mel spectrogram dimensions from the data
            
            # Apply each max pooling operation (each divides dimensions by 2)
            # We have 4 max pooling layers in our conv2Dblock
            h = h // (2**4)  # Divide by 2^4 for 4 layers with stride 2
            w = w // (2**4)
            
            # Final output channels is self.conv4_filters
            # Total flattened size is h * w * channels
            return h * w * self.conv4_filters
        except Exception as e:
            print(f"Error calculating CNN output dimensions: {e}")
            # Return a reasonable default
            return 256
    def forward(self, x):
        # Conv embedding
        conv_embedding = self.conv2Dblock(x)  # (b, channel, freq, time)
        batch_size = conv_embedding.size(0)
        
        # Store the shapes for debugging on first run
        if not hasattr(self, 'shapes_logged'):
            print(f"CNN output shape: {conv_embedding.shape}")
            self.shapes_logged = True
            
        conv_embedding = torch.flatten(conv_embedding, start_dim=1)  # flatten all except batch dimension
        
        if hasattr(self, 'shapes_logged') and self.shapes_logged:
            print(f"Flattened CNN shape: {conv_embedding.shape}")
        
        # Transformer embedding
        x_reduced = self.transf_maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)  # Remove channel dimension
        
        # Reshape input to match transformer dimensions
        batch_size, h, w = x_reduced.size()
        
        if hasattr(self, 'shapes_logged') and self.shapes_logged:
            print(f"Reduced input shape: {x_reduced.shape}")
        
        # Create a persistent projection layer instead of a transient one
        # First transpose (b, h, w) -> (b, w, h)
        x_transposed = x_reduced.transpose(1, 2)
        
        # Create a persistent projection layer if it doesn't exist
        if not hasattr(self, 'feature_projection'):
            self.feature_projection = nn.Linear(h, self.transf_dim).to(x_reduced.device)
            
        x_proj = self.feature_projection(x_transposed)  # (batch, seq_len, d_model)
        
        # Permute to transformer format: (seq_len, batch, d_model)
        x_proj = x_proj.permute(1, 0, 2)
        
        if hasattr(self, 'shapes_logged') and self.shapes_logged:
            print(f"Transformer input shape: {x_proj.shape}")
        
        transf_out = self.transf_encoder(x_proj)
        
        if hasattr(self, 'shapes_logged') and self.shapes_logged:
            print(f"Transformer output shape: {transf_out.shape}")
        
        # Average over sequence dimension to get (batch, d_model)
        transf_embedding = torch.mean(transf_out, dim=0)
        
        if hasattr(self, 'shapes_logged') and self.shapes_logged:
            print(f"Transformer embedding shape: {transf_embedding.shape}")        # Reshape CNN embedding if needed - use a persistent layer for dimension reduction
        if conv_embedding.size(1) > 10000:  # If CNN output is too large, apply dimensionality reduction
            if not hasattr(self, 'cnn_projection'):
                print("CNN embedding is too large, applying dimensionality reduction")
                print(f"Creating CNN projection from {conv_embedding.size(1)} to 256")
                self.cnn_projection = nn.Linear(conv_embedding.size(1), 256).to(x.device)
            conv_embedding = self.cnn_projection(conv_embedding)
            
        # If this is the first forward pass, we need to ensure the linear layer has the right dimensions
        if not hasattr(self, 'first_forward_done'):
            self.first_forward_done = True  # Mark the first forward pass as done
            
            # On first run, create CNN projection if needed but don't modify existing embedding
            if not hasattr(self, 'cnn_projection') and (self.cnn_flatten_dim != conv_embedding.size(1)):
                print(f"Creating CNN projection from {conv_embedding.size(1)} to {256}")
                self.cnn_projection = nn.Linear(conv_embedding.size(1), 256).to(x.device)
                conv_embedding = self.cnn_projection(conv_embedding)
        
        # Ensure transformer embedding has the right shape - use a persistent layer
        if transf_embedding.size(1) != self.transf_dim:
            if not hasattr(self, 'transf_proj'):
                print(f"Adjusting transformer embedding from {transf_embedding.size(1)} to {self.transf_dim}")
                self.transf_proj = nn.Linear(transf_embedding.size(1), self.transf_dim).to(x.device)
            transf_embedding = self.transf_proj(transf_embedding)
        
        if hasattr(self, 'shapes_logged') and self.shapes_logged:
            print(f"Final CNN shape: {conv_embedding.shape}")
            print(f"Final transformer shape: {transf_embedding.shape}")
        
        # Concatenate embeddings
        complete_embedding = torch.cat([conv_embedding, transf_embedding], dim=1)
        
        if hasattr(self, 'shapes_logged') and self.shapes_logged:
            print(f"Concatenated shape: {complete_embedding.shape}")
          # Adjust linear layer size if needed
        expected_input_size = conv_embedding.size(1) + transf_embedding.size(1)
        
        # Check if we need to recreate the linear layer
        if not hasattr(self, 'out_linear') or self.out_linear.in_features != expected_input_size:
            print(f"Creating/adjusting linear layer to input size: {expected_input_size}")
            
            # Get output dimension (num_emotions)
            out_dim = self.out_linear.out_features if hasattr(self, 'out_linear') else 4
            
            # Create a new linear layer with the correct input size
            new_linear = nn.Linear(expected_input_size, out_dim).to(x.device)
            
            # Transfer weights if possible (only if old linear layer exists)
            if hasattr(self, 'out_linear'):
                min_dim = min(self.out_linear.in_features, expected_input_size)
                with torch.no_grad():
                    new_linear.weight[:, :min_dim] = self.out_linear.weight[:, :min_dim]
                    new_linear.bias[:] = self.out_linear.bias[:]
            
            # Replace the old linear layer
            self.out_linear = new_linear
        
        # Final linear layer with softmax
        output_logits = self.out_linear(self.dropout_linear(complete_embedding))
        output_softmax = self.out_softmax(output_logits)
        
        return output_logits, output_softmax

# Training and validation functions
def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions, target=targets)

def make_train_step(model, loss_fnc, optimizer):
    def train_step(X, Y):
        # Set model to train mode
        model.train()
        # Forward pass
        output_logits, output_softmax = model(X)
        predictions = torch.argmax(output_softmax, dim=1)
        accuracy = torch.sum(Y == predictions) / float(len(Y))
        # Compute loss
        loss = loss_fnc(output_logits, Y)
        # Compute gradients
        loss.backward()
        # Update parameters and zero gradients
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), accuracy * 100
    return train_step

def make_validate_fnc(model, loss_fnc):
    def validate(X, Y):
        with torch.no_grad():
            model.eval()
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax, dim=1)
            accuracy = torch.sum(Y == predictions) / float(len(Y))
            loss = loss_fnc(output_logits, Y)
        return loss.item(), accuracy * 100, predictions
    return validate

# Function to create and train CNN model with given parameters
def train_cnn_with_params(params, X_train, Y_train, X_val, Y_val, device, epochs=50, batch_size=32):
    print(f"Training CNN with params: {params}")
    
    # Always use 4 emotions (our remapped emotions are 0, 1, 2, 3)
    num_emotions = len(EMOTION_NAMES)
    
    # Create model with parameters
    model = ParallelModel(num_emotions=num_emotions, params=params).to(device)
    
    # Define optimizer (using params for learning rate)
    lr = params.get('learning_rate', 0.01)
    weight_decay = params.get('weight_decay', 1e-3)
    momentum = params.get('momentum', 0.9)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    # Create training and validation functions
    train_step = make_train_step(model, loss_fnc, optimizer=optimizer)
    validate = make_validate_fnc(model, loss_fnc)
      # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, device=device).float()
    Y_train_tensor = torch.tensor(Y_train, device=device, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, device=device).float()
    Y_val_tensor = torch.tensor(Y_val, device=device, dtype=torch.long)
    
    # Train the model for a limited number of epochs
    dataset_size = len(X_train)
    losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Using sequential data without shuffling (as requested)
        X_train_epoch = X_train_tensor
        Y_train_epoch = Y_train_tensor
        
        epoch_loss = 0
        epoch_acc = 0
        iters = int(dataset_size / batch_size)
        
        for i in range(iters):
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, dataset_size)
            actual_batch_size = batch_end - batch_start
            
            X_batch = X_train_epoch[batch_start:batch_end]
            Y_batch = Y_train_epoch[batch_start:batch_end]
            
            loss, acc = train_step(X_batch, Y_batch)
            
            epoch_loss += loss * actual_batch_size / dataset_size
            epoch_acc += acc * actual_batch_size / dataset_size
        
        # Validate
        val_loss, val_acc, _ = validate(X_val_tensor, Y_val_tensor)
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}: train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.2f}%, val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")
    
    # Evaluate final validation accuracy
    _, final_val_acc, _ = validate(X_val_tensor, Y_val_tensor)
    
    return model, final_val_acc

# Objective function for EO to optimize CNN parameters
def fitness_function(solution):
    """Fitness function for Equilibrium Optimizer to minimize."""    
    # Map solution vector to CNN parameters
    # Calculate number of heads first - we'll use fixed heads for optimization
    transf_heads = 6  # Using 6 heads because that's what was seen in the error
    
    # Make sure transformer dimension is divisible by number of heads
    # Base dimension = 32, max multiplier = 8 (32*8 = 256)
    transf_dim_base = 32
    transf_dim_max_multiplier = 8
    transf_dim_multiplier = int(1 + solution[8] * (transf_dim_max_multiplier - 1))
    
    # Calculate dimension and explicitly ensure divisibility
    transf_dim_raw = transf_dim_base * transf_dim_multiplier
    transf_dim = (transf_dim_raw // transf_heads) * transf_heads
    
    # Make sure dimension is at least equal to number of heads
    if transf_dim < transf_heads:
        transf_dim = transf_heads  # Minimum dimension = number of heads
    
    # Debug information
    print(f"Transformer heads: {transf_heads}, raw dimension: {transf_dim_raw}, adjusted dimension: {transf_dim}")
    
    params = {
        'conv1_filters': int(16 * (1 + solution[0] * 3)),  # 16-64 filters
        'conv2_filters': int(32 * (1 + solution[1] * 3)),  # 32-128 filters
        'conv3_filters': int(64 * (1 + solution[2] * 3)),  # 64-256 filters
        'conv4_filters': int(64 * (1 + solution[3] * 3)),  # 64-256 filters
        'kernel_size': int(2 + solution[4] * 3),  # 2-5 kernel size
        'dropout_rate': 0.2 + solution[5] * 0.4,  # 0.2-0.6 dropout
        'transf_heads': transf_heads,  # 1-8 heads
        'transf_layers': 4,  # Fixed at exactly 4 layers as requested
        'transf_dim': transf_dim,  # Ensuring divisibility by heads
        'learning_rate': 0.001 + solution[9] * 0.049,  # 0.001-0.05
        'weight_decay': 0.0001 + solution[10] * 0.005,  # 0.0001-0.005
        'momentum': 0.7 + solution[11] * 0.25,  # 0.7-0.95
    }
      # Very short training to assess parameter quality faster
    model, val_acc = train_cnn_with_params(
        params, 
        X_train_global, 
        Y_train_global, 
        X_val_global, 
        Y_val_global,
        device=device_global,
        epochs=2,  # Extremely limited epochs for faster optimization
        batch_size=64  # Larger batch size for faster iteration
    )
    
    # Convert accuracy to fitness (lower is better)
    fitness = 100 - val_acc
    
    # Add a small penalty for model complexity
    num_params = sum(p.numel() for p in model.parameters())
    normalized_params = num_params / 1_000_000  # Normalize by million
    complexity_penalty = 0.1 * normalized_params
      # Final fitness - ensure it's a single float value
    fitness = float(fitness + complexity_penalty)
    
    print(f"Solution fitness: {fitness:.4f} (val_acc: {val_acc:.2f}%, complexity: {normalized_params:.2f}M params)")
    
    # Ensure we're returning a single float value
    return float(fitness)

# Start from here - Split and preprocess the data
print("Preparing to shuffle and split the data...")
print("Data shape before shuffling:", mel_spectrograms.shape)
print("Number of emotion labels:", len(Data['Emotion']))

# Verify both arrays have the same length
if len(mel_spectrograms) != len(Data):
    print(f"WARNING: Length mismatch between features ({len(mel_spectrograms)}) and labels ({len(Data)})")
    print("Adjusting Data DataFrame to match mel_spectrograms length...")
    # Keep only the first len(mel_spectrograms) entries in Data
    if len(mel_spectrograms) < len(Data):
        Data = Data.iloc[:len(mel_spectrograms)]
    else:
        # If we have more spectrograms than labels, trim the spectrograms
        mel_spectrograms = mel_spectrograms[:len(Data)]
    print(f"After adjustment: mel_spectrograms shape: {mel_spectrograms.shape}, Data length: {len(Data)}")

# Do not shuffle the data before splitting (as requested)
print("Using sequential order without shuffling (as requested)...")
try:
    # Keep original order
    original_emotions = Data['Emotion'].values
      # Map the original emotion indices to consecutive indices (0, 1, 2, 3)
    remapped_emotions = np.array([EMOTION_MAPPING.get(emo, 0) for emo in original_emotions])
    
    X = mel_spectrograms
    y = remapped_emotions
    
    print("Data kept in sequential order and emotions remapped")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(f"Original emotion distribution: {np.unique(original_emotions, return_counts=True)}")
    print(f"Remapped emotion distribution: {np.unique(remapped_emotions, return_counts=True)}")
except Exception as e:
    print(f"ERROR during data shuffling: {e}")
    # Try to continue without shuffling
    print("Attempting to continue without shuffling...")
    X = mel_spectrograms
    original_emotions = Data['Emotion'].values
    # Map the original emotion indices to consecutive indices (0, 1, 2, 3)
    y = np.array([EMOTION_MAPPING.get(emo, 0) for emo in original_emotions])

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

# Normalize the data
scaler = StandardScaler()

# Reshape for scaling (flatten non-batch dimensions)
b, c, h, w = X_train.shape
X_train_flat = X_train.reshape(b, -1)
X_train_flat = scaler.fit_transform(X_train_flat)
X_train = X_train_flat.reshape(b, c, h, w)

b, c, h, w = X_val.shape
X_val_flat = X_val.reshape(b, -1)
X_val_flat = scaler.transform(X_val_flat)
X_val = X_val_flat.reshape(b, c, h, w)

b, c, h, w = X_test.shape
X_test_flat = X_test.reshape(b, -1)
X_test_flat = scaler.transform(X_test_flat)
X_test = X_test_flat.reshape(b, c, h, w)

# Free up memory
del X, mel_spectrograms, X_train_flat, X_val_flat, X_test_flat
gc.collect()

# Set up global variables for EO optimization
device_global = 'cuda' if torch.cuda.is_available() else 'cpu'
X_train_global = X_train
Y_train_global = y_train
X_val_global = X_val
Y_val_global = y_val
print(f"Using device: {device_global}")

# Hyperparameter optimization with EO
if not SKIP_TRAINING and not os.path.exists(INDICES_CACHE_PATH_EO):
    print("\n" + "="*50)
    print("OPTIMIZING CNN HYPERPARAMETERS WITH EQUILIBRIUM OPTIMIZER")
    print("="*50)    # Define parameters for EO
    problem_size = 12  # Number of parameters to optimize
    # Use FloatVar from mealpy.utils.space
    from mealpy.utils.space import FloatVar
    bounds = [FloatVar(lb=0.0, ub=1.0) for _ in range(problem_size)]  # Bounds for each parameter [min, max]
    
    problem_dict = {
        "obj_func": fitness_function,
        "bounds": bounds,
        "minmax": "min",
        "log_to": "file",
        "log_file": "eo_cnn_convergence.log",
        "verbose": True
    }
    
    print("Starting EO optimization...")
    start_time = time.time()    # Create EO model with minimal parameters to finish in ~30 minutes
    model_eo = EO.OriginalEO(epoch=3, pop_size=5)  # Minimum allowed pop_size is 5
    best_solution = model_eo.solve(problem_dict)
    
    end_time = time.time()
    print(f"EO optimization finished in {end_time - start_time:.2f} seconds.")
    
    # Extract the best hyperparameters
    best_position = best_solution.solution
    best_fitness = best_solution.target.fitness
    
    print(f"Best fitness found: {best_fitness:.4f}")
    print(f"Best hyperparameters: {best_position}")
    
    # Save the EO history and best hyperparameters
    try:
        eo_history = np.array(model_eo.history.list_global_best_fit)
        np.save(EO_HISTORY_PATH, eo_history)
        np.save(INDICES_CACHE_PATH_EO, best_position)
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plt.plot(eo_history, 'b-')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value (lower is better)')
        plt.title('EO Convergence for CNN Hyperparameters')
        plt.grid(True)
        plt.savefig('eo_cnn_convergence.png')
        plt.close()
    except Exception as e:
        print(f"Could not extract or plot convergence history: {e}")
      # Map best solution to CNN parameters
    # Calculate number of heads first
    transf_heads = int(1 + best_position[6] * 7)
    
    # Calculate transformer dimension with explicit handling for divisibility by heads
    transf_dim_base = 32
    transf_dim_multiplier = int(1 + best_position[8] * 7)
    transf_dim_raw = transf_dim_base * transf_dim_multiplier
    # Adjust to make divisible by number of heads
    transf_dim = (transf_dim_raw // transf_heads) * transf_heads
    if transf_dim < transf_heads:  # Make sure it's at least as large as the number of heads
        transf_dim = transf_heads
        
    best_params = {
        'conv1_filters': int(16 * (1 + best_position[0] * 3)),
        'conv2_filters': int(32 * (1 + best_position[1] * 3)),
        'conv3_filters': int(64 * (1 + best_position[2] * 3)),
        'conv4_filters': int(64 * (1 + best_position[3] * 3)),
        'kernel_size': int(2 + best_position[4] * 3),
        'dropout_rate': 0.2 + best_position[5] * 0.4,
        'transf_heads': transf_heads,
        'transf_layers': 4,  # Fixed at 4 layers as requested
        'transf_dim': transf_dim,  # Adjusted to be divisible by heads
        'learning_rate': 0.001 + best_position[9] * 0.049,
        'weight_decay': 0.0001 + best_position[10] * 0.005,
        'momentum': 0.7 + best_position[11] * 0.25,
    }
    
    print(f"Best parameters mapped: {best_params}")
    
elif os.path.exists(INDICES_CACHE_PATH_EO):
    # Load cached best hyperparameters
    best_position = np.load(INDICES_CACHE_PATH_EO)
    
    # Calculate number of heads first
    transf_heads = int(1 + best_position[6] * 7)
    
    # Calculate transformer dimension with explicit handling for divisibility by heads
    transf_dim_base = 32
    transf_dim_multiplier = int(1 + best_position[8] * 7)
    transf_dim_raw = transf_dim_base * transf_dim_multiplier
    # Adjust to make divisible by number of heads
    transf_dim = (transf_dim_raw // transf_heads) * transf_heads
    if transf_dim < transf_heads:  # Make sure it's at least as large as the number of heads
        transf_dim = transf_heads
      
    # Map best solution to CNN parameters
    best_params = {
        'conv1_filters': int(16 * (1 + best_position[0] * 3)),
        'conv2_filters': int(32 * (1 + best_position[1] * 3)),
        'conv3_filters': int(64 * (1 + best_position[2] * 3)),
        'conv4_filters': int(64 * (1 + best_position[3] * 3)),
        'kernel_size': int(2 + best_position[4] * 3),
        'dropout_rate': 0.2 + best_position[5] * 0.4,
        'transf_heads': transf_heads,
        'transf_layers': 4,  # Fixed at 4 layers as requested
        'transf_dim': transf_dim,  # Adjusted to be divisible by heads
        'learning_rate': 0.001 + best_position[9] * 0.049,
        'weight_decay': 0.0001 + best_position[10] * 0.005,
        'momentum': 0.7 + best_position[11] * 0.25,
    }
    
    print(f"Loaded cached best parameters: {best_params}")
    
else:
    # Default parameters if no optimization was done
    transf_heads = 4
    transf_dim = 64  # Ensure divisibility: 64 ÷ 4 = 16
    
    best_params = {
        'conv1_filters': 16,
        'conv2_filters': 32,
        'conv3_filters': 64,
        'conv4_filters': 64,
        'kernel_size': 3,
        'dropout_rate': 0.3,
        'transf_heads': transf_heads,
        'transf_layers': 4,
        'transf_dim': transf_dim,
        'learning_rate': 0.01,
        'weight_decay': 0.001,
        'momentum': 0.8,
    }
    
    print(f"Using default parameters: {best_params}")

# Train the final model with best parameters
if not SKIP_TRAINING:
    print("\n" + "="*50)
    print("TRAINING FINAL CNN MODEL WITH OPTIMIZED PARAMETERS")
    print("="*50)
      # Train for fewer epochs to complete in approximately 30 minutes
    device = device_global
    num_emotions = len(EMOTIONS)
    epochs = 20  # Reduced from 100 to 20
    batch_size = 64  # Increased from 32 to 64 for faster iteration
    
    # Create model with optimized parameters
    final_model = ParallelModel(num_emotions=num_emotions, params=best_params).to(device)
    
    # Define optimizer with optimized hyperparameters
    lr = best_params.get('learning_rate')
    weight_decay = best_params.get('weight_decay')
    momentum = best_params.get('momentum')
    optimizer = optim.SGD(final_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    # Create training and validation functions
    train_step = make_train_step(final_model, loss_fnc, optimizer=optimizer)
    validate = make_validate_fnc(final_model, loss_fnc)
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, device=device).float()
    Y_train_tensor = torch.tensor(y_train, device=device, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, device=device).float()
    Y_val_tensor = torch.tensor(y_val, device=device, dtype=torch.long)
    
    # Training loop
    dataset_size = len(X_train)
    losses = []
    val_losses = []
    best_val_acc = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Using sequential data without shuffling (as requested)
        X_train_epoch = X_train_tensor
        Y_train_epoch = Y_train_tensor
        
        epoch_loss = 0
        epoch_acc = 0
        iters = int(dataset_size / batch_size)
        
        for i in range(iters):
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, dataset_size)
            actual_batch_size = batch_end - batch_start
            
            X_batch = X_train_epoch[batch_start:batch_end]
            Y_batch = Y_train_epoch[batch_start:batch_end]
            
            loss, acc = train_step(X_batch, Y_batch)
            
            epoch_loss += loss * actual_batch_size / dataset_size
            epoch_acc += acc * actual_batch_size / dataset_size
            
            print(f"\rEpoch {epoch+1}/{epochs}: iteration {i+1}/{iters}", end="")
            
        # Validate
        val_loss, val_acc, _ = validate(X_val_tensor, Y_val_tensor)
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(final_model.state_dict(), MODEL_CACHE_PATH)
            print(f"\nSaved new best model at epoch {epoch+1} with val_acc: {val_acc:.2f}%")
        
        print(f"\nEpoch {epoch+1}/{epochs}: train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.2f}%, val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")
        
    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(losses, 'b-', label='Training Loss')
    plt.plot(val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('cnn_eo_training_curve.png')
    plt.close()
    
else:
    # Load the saved model
    num_emotions = len(EMOTIONS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    final_model = ParallelModel(num_emotions=num_emotions, params=best_params).to(device)
    final_model.load_state_dict(torch.load(MODEL_CACHE_PATH, map_location=device))
    print(f"Loaded model from {MODEL_CACHE_PATH}")

# Evaluate the model on test data
print("\n" + "="*50)
print("EVALUATING MODEL ON TEST DATA")
print("="*50)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
final_model.eval()
validate = make_validate_fnc(final_model, loss_fnc)

X_test_tensor = torch.tensor(X_test, device=device).float()
Y_test_tensor = torch.tensor(y_test, device=device, dtype=torch.long)

# Get predictions on test set
test_loss, test_acc, predictions = validate(X_test_tensor, Y_test_tensor)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.2f}%")

# Convert predictions to numpy for analysis
predictions = predictions.cpu().numpy()
y_test_cpu = y_test

# Calculate more detailed metrics
unique_emotions = sorted(np.unique(np.concatenate([y_test_cpu, predictions])))
emotion_names = [EMOTION_NAMES[e] for e in unique_emotions]

# Confusion matrix
cm = confusion_matrix(y_test_cpu, predictions, labels=unique_emotions)

# Classification report
report = classification_report(
    y_test_cpu,
    predictions,
    labels=unique_emotions,
    target_names=emotion_names,
    output_dict=True
)

# F1 score, precision, recall
test_f1 = f1_score(y_test_cpu, predictions, average='weighted')
test_precision = precision_score(y_test_cpu, predictions, average='weighted')
test_recall = recall_score(y_test_cpu, predictions, average='weighted')

print(f"Test F1 Score (Weighted): {test_f1:.4f}")
print(f"Test Precision (Weighted): {test_precision:.4f}")
print(f"Test Recall (Weighted): {test_recall:.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=emotion_names,
    yticklabels=emotion_names
)
plt.title('Confusion Matrix (Parallel CNN with EO)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix_cnn_eo.png')
plt.close()

# Per-class accuracy
class_accuracies = {}
for i, emotion in enumerate(unique_emotions):
    class_accuracies[emotion] = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0

# Plot per-class accuracy
plt.figure(figsize=(10, 6))
emotion_list = [EMOTIONS.get(e, f"Unknown-{e}") for e in unique_emotions]
accuracy_list = [class_accuracies[e] * 100 for e in unique_emotions]

plt.bar(emotion_list, accuracy_list)
plt.xlabel('Emotion')
plt.ylabel('Accuracy (%)')
plt.title('Per-Class Accuracy (Parallel CNN with EO)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('accuracy_by_emotion_cnn_eo.png')
plt.close()

# Save evaluation results
results = {
    'test_accuracy': test_acc,
    'test_f1': test_f1,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'confusion_matrix': cm,
    'classification_report': report,
    'class_accuracies': class_accuracies,
    'best_params': best_params
}

joblib.dump(results, EVALUATION_RESULTS_PATH)
print(f"Evaluation results saved to {EVALUATION_RESULTS_PATH}")

# Generate a markdown report
report_content = f"""
# Arabic Emotion Recognition Model Evaluation Report (Parallel CNN with EO)

## Model Architecture
- Parallel CNN + Transformer architecture
- CNN parameters optimized using Equilibrium Optimizer (EO)

## Optimized Hyperparameters
```
{best_params}
```

## Overall Performance
- Test Accuracy: {test_acc:.2f}%
- Test F1 Score (Weighted): {test_f1:.4f}
- Test Precision (Weighted): {test_precision:.4f}
- Test Recall (Weighted): {test_recall:.4f}

## Per-Class Accuracy
"""

for emotion in unique_emotions:
    emotion_name = EMOTIONS.get(emotion, f"Unknown-{emotion}")
    report_content += f"- {emotion_name}: {class_accuracies[emotion]*100:.2f}%\n"

with open('evaluation_report_cnn_eo.md', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("Evaluation report saved to evaluation_report_cnn_eo.md")
print("\nArabic Emotion Recognition with Parallel CNN + EO complete!")

# Command-line interface for running the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arabic Emotion Recognition with Parallel CNN and EO')
    parser.add_argument('--clear-all', action='store_true', 
                        help='Clear all cached files before running')
    parser.add_argument('--clear-model', action='store_true', 
                        help='Clear only model files before running')
    args = parser.parse_args()
    
    # Handle cache clearing if requested
    if args.clear_all:
        cache_files = [
            FEATURES_CACHE_PATH, DATA_CACHE_PATH, INDICES_CACHE_PATH_EO,
            MODEL_CACHE_PATH, EVALUATION_RESULTS_PATH, EO_HISTORY_PATH
        ]
        for file in cache_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Deleted cache file: {file}")
    
    elif args.clear_model:
        model_files = [
            INDICES_CACHE_PATH_EO, MODEL_CACHE_PATH, EVALUATION_RESULTS_PATH
        ]
        for file in model_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Deleted model file: {file}")
                
    print("Parallel CNN with EO hyperparameter optimization complete.")
