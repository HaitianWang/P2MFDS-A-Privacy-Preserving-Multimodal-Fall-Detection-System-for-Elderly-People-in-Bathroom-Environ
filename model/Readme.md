# Multimodal Fall Detection

This folder contains deep learning models for fall detection using millimeter-wave radar and vibration sensors.

## Contents

- **Multimodal CNN-LSTM with Attention**: Combines radar and vibration sensor inputs. Each input is processed by independent CNN and LSTM stream followed by attention modules, then fused for final classification.
- **CNN, LSTM, CNN-LSTM Models**: Applies different deep learning architectures to radar data only for comparative analysis.

## Model Descriptions

### Multimodal CNN-LSTM with Attention
A dual-stream model that processes radar and vibration signals separately using:
- CNN for spatial feature extraction
- LSTM for temporal modeling
- Attention for weighting relevant time steps
- Feature fusion and fully connected layers for binary classification

### CNN, LSTM, and CNN-LSTM
- **CNN**: Captures local patterns across the radar signal sequence
- **LSTM**: Learns long-term temporal dependencies
- **CNN-LSTM**: Combines both local and temporal learning for improved performance

## Key Features

- Multimodal fusion for higher accuracy
- Temporal attention mechanism
- Lightweight architecture
- Comparative benchmarking of multiple models

## Evaluation Metrics

- Accuracy
- ROC Curve and AUC
- Precision, Recall, F1-score
- Confusion Matrix

## Usage

Run the multimodal model:
```bash
python millimeter-wave radar and three-axis vibration sensors 2.py
```
Run the radar-only models:
```bash
python nnWave_LSTM_prediction.py
```
