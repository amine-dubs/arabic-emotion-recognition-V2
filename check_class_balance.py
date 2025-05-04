import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to the cached data
DATA_CACHE_PATH = 'data_df.pkl'

print("Checking class balance in the dataset...")

if os.path.exists(DATA_CACHE_PATH):
    # Load the DataFrame
    data_df = pd.read_pickle(DATA_CACHE_PATH)
    
    # Map emotion codes to names
    EMOTIONS = { 0 : 'Angry',
                 2 : 'Happy',
                 3 : 'Neutral',
                 4 : 'Sad'
               }
    
    # Get the emotion counts
    emotion_counts = data_df['Emotion'].value_counts().sort_index()
    
    # Print the raw counts
    print("\nRaw emotion counts:")
    for emotion_code, count in emotion_counts.items():
        emotion_name = EMOTIONS.get(emotion_code, f"Unknown-{emotion_code}")
        print(f"{emotion_name}: {count} samples")
    
    # Calculate percentages
    total_samples = len(data_df)
    print(f"\nTotal samples: {total_samples}")
    
    emotion_percentages = (emotion_counts / total_samples * 100).round(1)
    print("\nPercentage distribution:")
    for emotion_code, percentage in emotion_percentages.items():
        emotion_name = EMOTIONS.get(emotion_code, f"Unknown-{emotion_code}")
        print(f"{emotion_name}: {percentage}%")
    
    # Generate a bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        [EMOTIONS.get(code, f"Unknown-{code}") for code in emotion_counts.index], 
        emotion_counts.values,
        color=['#FF7F7F', '#FFBF7F', '#FFDF7F', '#FFFF7F']
    )
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 5,
            f'{int(height)}',
            ha='center', 
            va='bottom'
        )
    
    plt.title('Emotion Class Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Samples')
    plt.savefig('class_distribution.png')
    print("\nSaved class distribution plot to 'class_distribution.png'")
    
    # Check class balance in training/test splits
    print("\nChecking balance in train/test splits (if available)...")
    if 'Split' in data_df.columns:
        print("Split column found. Analyzing train/test distribution...")
        
        # Count by split and emotion
        split_emotion_counts = data_df.groupby(['Split', 'Emotion']).size().unstack(fill_value=0)
        print(split_emotion_counts)
    else:
        print("No split column found in the DataFrame.")
        
        # Check for balance across actors
        if 'Path' in data_df.columns:
            # Extract actor information from file paths
            data_df['Actor'] = data_df['Path'].str.extract(r'\\([^\\]+)\\')
            
            # Count samples by actor and emotion
            actor_emotion_counts = data_df.groupby(['Actor', 'Emotion']).size().unstack(fill_value=0)
            print("\nSample counts by actor and emotion:")
            print(actor_emotion_counts)
            
            # Plot distribution for each actor
            plt.figure(figsize=(12, 8))
            actor_emotion_counts.plot(kind='bar', ax=plt.gca())
            plt.title('Emotion Distribution by Actor')
            plt.xlabel('Actor')
            plt.ylabel('Number of Samples')
            plt.tight_layout()
            plt.savefig('actor_emotion_distribution.png')
            print("Saved actor emotion distribution to 'actor_emotion_distribution.png'")
else:
    print(f"Error: Data file not found at {DATA_CACHE_PATH}")