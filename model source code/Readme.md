# Multimodal Fall Detection

This folder contains deep learning models for fall detection using millimeter-wave radar and vibration sensors.

## Contents

- **Multimodal CNN-LSTM with Attention**: Combines radar and vibration sensor inputs. Each input is processed by independent CNN and LSTM stream followed by attention modules, then fused for final classification.
- **CNN, LSTM, CNN-LSTM Models**: Applies different deep learning architectures to radar data only for ablation study.

## Model Descriptions

### Multimodal CNN-LSTM with Attention + SEB
This model follows a two-stream architecture:

- **Radar stream**: The radar signal passes through two 1D convolutional layers followed by a bidirectional LSTM. An attention mechanism is applied to emphasize important temporal features.
- **Vibration stream**: The vibration data follows the same structure as the radar stream (CNN → Bi-LSTM → Attention).
- **SEB (Selective Enhancement Block)**: A custom-designed attention block that adaptively enhances discriminative features across modalities before fusion.
- **Fusion**: Features from both streams are refined by SEB and then passed through fully connected layers for binary classification (fall vs. no fall).

This architecture allows the model to learn both spatial and temporal patterns from each modality independently and enhance cross-modal interactions via SEB.

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
python millimeter-wave radar and three-axis vibration sensors.py
```
Run the radar-only models:
```bash
python nnWave_LSTM_prediction.py
```
