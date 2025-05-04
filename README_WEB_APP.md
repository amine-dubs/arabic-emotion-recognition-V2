# Arabic Audio Emotion Recognition Web Application

This web application provides a user-friendly interface for the Arabic audio emotion recognition system, allowing users to:

1. Upload audio files for emotion analysis
2. View visualizations of audio spectrograms
3. Get emotion predictions with confidence scores
4. Receive personalized recommendations on how to respond to detected emotions

## Features

- **Modern UI/UX**: Clean, responsive interface built with Streamlit
- **Audio Visualization**: Display mel-spectrograms of uploaded audio
- **Emotion Classification**: Detect emotions including Angry, Happy, Neutral, and Sad
- **Personalized Response Recommendations**: Get advice on how to respond based on speaker relationship, age, gender, and context
- **Arabic Language Support**: Optimized for Arabic audio recordings
- **Model Evaluation**: View current model performance metrics (training accuracy ~51%, test accuracy ~41%)

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- A Groq API key (for response recommendations)

### Installation Steps

1. **Clone the repository** (if applicable):

```bash
git clone https://github.com/your-username/arabic-emotion-recognition.git
cd arabic-emotion-recognition
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Set up the Groq API key** (for recommendation feature):

   - Register for an API key at [Groq](https://console.groq.com/)
   - Set the environment variable:

     ```bash
     # On Windows
     set GROQ_API_KEY=your_api_key_here

     # On macOS/Linux
     export GROQ_API_KEY=your_api_key_here
     ```

4. **Run the application**:

```bash
streamlit run app.py
```

5. **Access the web interface** in your browser at `http://localhost:8501`

## Usage

1. **Upload an audio file**: 
   - Click on "Browse files" in the upload section
   - Select a WAV, MP3, or OGG file (preferably containing Arabic speech)

2. **Listen to the audio** using the embedded audio player

3. **Provide contextual information** in the sidebar:
   - Relationship to the speaker
   - Age group
   - Gender
   - Context of the conversation

4. **Analyze emotion** by clicking one of the "Analyze Emotion" buttons:
   - "Analyze Emotion (with scaling)" - Uses feature scaling
   - "Analyze Emotion (without scaling)" - Bypasses feature scaling

5. **View results**:
   - Mel-spectrogram visualization
   - Detected emotion with icon
   - Probability distribution (if available)

6. **Get response recommendations** by clicking "Generate Response Recommendations"

## How It Works

The application processes audio through a pipeline:
1. **Audio Processing**: Converts audio to mel-spectrograms
2. **Feature Extraction**: Uses ResNet50 to extract deep features
3. **Feature Selection**: Applies pre-selected features identified by Sine Cosine Algorithm (SCA)
4. **Classification**: Predicts emotions using a pre-trained k-Nearest Neighbors model
5. **Response Generation**: Uses Groq's LLM to generate personalized recommendations

## Data and Model Details

- **Data Split**: 80% training, 20% test (no separate validation set)
- **Current Model Performance**:
  - Training Accuracy: ~51%
  - Test Accuracy: ~41%
  - Test F1 Score (Weighted): ~41%
- **Feature Selection**: SCA optimized feature selection using test accuracy in fitness function
- **Available Processing Methods**:
  - With feature scaling (generally better performance)
  - Without feature scaling (alternative approach for comparison)

## Retraining Options

The application provides several retraining options accessible from the Training Results tab:

1. **Quick retrain**: Keeps features and data, only retrains the model (fastest)
2. **Full retune**: Keeps features and data, runs feature selection and model training
3. **Complete retraining**: Starts from scratch with all processing steps

## Limitations

- Best results with high-quality audio recordings
- Optimized for short audio clips (3 seconds)
- May not perform well with background noise or multiple speakers
- Response recommendations require a valid Groq API key
- Current model has moderate accuracy (~41% on test set)

## Troubleshooting

- **Models not loading**: Ensure all model files (scaler.joblib, knn_final_model.joblib, selected_indices.npy) are in the application directory
- **Audio processing errors**: Try with a different audio file or ensure the audio contains clear speech
- **GROQ API errors**: Verify your API key is correctly set as an environment variable
- **Low accuracy**: Current model training may need improvement; consider using the retraining options with enhanced parameters

## License

This project is licensed under the [MIT License](LICENSE).