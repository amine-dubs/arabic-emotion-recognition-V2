
# Arabic Emotion Recognition Model Evaluation Report (Parallel CNN with EO)

## Model Architecture
- Parallel CNN + Transformer architecture
- CNN parameters optimized using Equilibrium Optimizer (EO)

## Optimized Hyperparameters
```
{'conv1_filters': 53, 'conv2_filters': 32, 'conv3_filters': 122, 'conv4_filters': 152, 'kernel_size': 5, 'dropout_rate': 0.28892019290035714, 'transf_heads': 6, 'transf_layers': 4, 'transf_dim': 156, 'learning_rate': 0.03246403163066978, 'weight_decay': 0.003019192486162942, 'momentum': 0.7049893908374005}
```

## Overall Performance
- Test Accuracy: 80.84%
- Test F1 Score (Weighted): 0.8140
- Test Precision (Weighted): 0.8442
- Test Recall (Weighted): 0.8084

## Per-Class Accuracy
- Angry: 77.94%
- Unknown-1: 81.36%
- Happy: 73.53%
- Neutral: 90.91%
