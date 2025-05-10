# Arabic Emotion Recognition using Parallel CNN-Transformer with Equilibrium Optimizer

This project implements an Arabic speech emotion recognition system using a hybrid Parallel CNN-Transformer architecture with hyperparameter optimization via the Equilibrium Optimizer (EO) algorithm. The model has been optimized to run in approximately 30 minutes with sequential data processing (no shuffling).

## Architecture Overview

The model employs a parallel architecture that combines the strengths of:

1. **Convolutional Neural Networks (CNNs)** - For efficient feature extraction from mel-spectrogram representations of audio
2. **Transformer Encoders** - For capturing sequential dependencies in audio features
3. **Equilibrium Optimizer** - For optimizing hyperparameters of the network

### Key Features

- Processes Arabic speech audio files from the EYASE dataset
- Extracts mel-spectrograms as input features
- Applies data augmentation using Additive White Gaussian Noise (AWGN)
- Uses a fixed 4-layer transformer architecture as requested
- Optimizes other hyperparameters including:
  - CNN filter counts (16-64, 32-128, 64-256, 64-256)
  - Kernel sizes (2-5)
  - Dropout rates (0.2-0.6)
  - Transformer heads (1-8)
  - Transformer dimensions
  - Learning rates, weight decay, and momentum

## Data Processing Pipeline

1. **Audio Loading**: Uses librosa to load audio files from the EYASE Arabic emotion dataset
2. **Feature Extraction**: Converts raw audio to mel-spectrograms with 128 mel bands
3. **Data Augmentation**: Applies AWGN to increase training data and improve model robustness
4. **Normalization**: Standardizes features using StandardScaler
5. **Data Splitting**: 70% training, 15% validation, 15% testing with stratified sampling

## Model Architecture

### Parallel CNN Component
- 4 convolutional blocks with increasing filter counts
- Each block contains: Conv2D → BatchNorm → ReLU → MaxPool → Dropout

### Transformer Component
- Fixed 4-layer transformer encoder
- Multi-head self-attention mechanism
- Variable number of attention heads (optimized via EO)
- Variable embedding dimensions (optimized via EO)

### Final Classification
- Concatenation of CNN and transformer embeddings
- Dropout layer for regularization
- Linear layer with softmax activation for 4-class classification (Angry, Happy, Neutral, Sad)

## Hyperparameter Optimization

The Equilibrium Optimizer (EO) is used to find optimal hyperparameters:

- Search space normalized to [0,1] for each parameter
- Parameters mapped to appropriate ranges during fitness evaluation
- Fitness function includes:
  - Validation accuracy (primary objective)
  - Model complexity penalty (to prevent overfitting)

## Usage

To train a new model:
```bash
python eaed-using-parallel-cnn-eo.py
```

To evaluate a pre-trained model:
```bash
python eaed-using-parallel-cnn-eo.py --eval-only
```

To clear cached model and train from scratch:
```bash
python eaed-using-parallel-cnn-eo.py --clear-model
```

## Results

The model achieves competitive emotion recognition accuracy on Arabic speech data while maintaining efficient inference time through the parallel architecture.

## Dependencies

- Python 3.x
- PyTorch
- Librosa (for audio processing)
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn (for visualization)
- MEALPY (for the Equilibrium Optimizer implementation)

## Notes

- The transformer component is fixed at exactly 4 layers as required
- Training uses sequential data processing (no shuffling) as requested
- Optimization time reduced to approximately 30 minutes with:
  - Minimal EO parameters (3 epochs, 3 population size)
  - Fixed transformer heads (4)
  - Reduced dimension sizes
  - Larger batch sizes
  - Fewer training epochs
- Caching is used extensively to speed up repeated runs
