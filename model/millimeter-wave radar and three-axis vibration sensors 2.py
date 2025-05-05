#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multimodal Fall Detection System with CNN-LSTM Architecture

This script implements a multimodal deep learning framework combining radar and vibration data
for human fall detection. The model integrates CNN, LSTM and attention mechanisms to process
time-series data from multiple sensors.
"""

# Standard library imports
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Constants
DEFAULT_SEQUENCE_LENGTH = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 0.0005
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Configure matplotlib for academic style
plt.style.use("seaborn-poster")
plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (10, 6),
    "figure.autolayout": True,
    "savefig.dpi": 300,
})

class MultimodalDataset(Dataset):
    """Custom dataset class for multimodal sensor data.
    
    Attributes:
        radar_data (torch.Tensor): Processed radar sequences
        vibration_data (torch.Tensor): Processed vibration sequences
        labels (torch.Tensor): Corresponding labels
    """
    
    def __init__(
        self,
        radar_data: torch.Tensor,
        vibration_data: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.radar_data = radar_data
        self.vibration_data = vibration_data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.radar_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.radar_data[idx],
            self.vibration_data[idx],
            self.labels[idx],
        )

class AttentionModule(nn.Module):
    """Attention mechanism for temporal feature weighting.
    
    Args:
        input_dim (int): Dimension of input features
    """
    
    def __init__(self, input_dim: int):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_weights = torch.softmax(self.attention(x), dim=1)
        return torch.sum(x * attn_weights, dim=1)

class MultimodalCNNLSTM(nn.Module):
    """Multimodal CNN-LSTM architecture with feature fusion.
    
    Args:
        radar_input_dim (int): Dimension of radar input features
        vibration_input_dim (int): Dimension of vibration input features
        num_classes (int): Number of output classes
    """
    
    def __init__(
        self,
        radar_input_dim: int = 3,
        vibration_input_dim: int = 3,
        num_classes: int = 1,
    ):
        super(MultimodalCNNLSTM, self).__init__()

        # Radar processing branch
        self.radar_cnn = nn.Sequential(
            nn.Conv1d(radar_input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.radar_lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.radar_attention = AttentionModule(64 * 2)

        # Vibration processing branch
        self.vibration_cnn = nn.Sequential(
            nn.Conv1d(vibration_input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.vibration_lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.vibration_attention = AttentionModule(64 * 2)

        # Feature fusion and classification
        self.fusion_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )

    def forward(
        self,
        radar_input: torch.Tensor,
        vibration_input: torch.Tensor,
    ) -> torch.Tensor:
        # Radar feature extraction
        radar_features = self.radar_cnn(radar_input.permute(0, 2, 1))
        radar_temporal, _ = self.radar_lstm(radar_features.permute(0, 2, 1))
        radar_attn = self.radar_attention(radar_temporal)

        # Vibration feature extraction
        vibration_features = self.vibration_cnn(vibration_input.permute(0, 2, 1))
        vibration_temporal, _ = self.vibration_lstm(vibration_features.permute(0, 2, 1))
        vibration_attn = self.vibration_attention(vibration_temporal)

        # Feature fusion
        fused_features = torch.cat([radar_attn, vibration_attn], dim=1)
        return self.fusion_fc(fused_features)

def load_and_preprocess_data(
    radar_path: str,
    vibration_path: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and preprocess multimodal sensor data.
    
    Args:
        radar_path (str): Path to radar data CSV
        vibration_path (str): Path to vibration data CSV
        sequence_length (int): Length of sliding window
    
    Returns:
        Tuple containing:
        - radar_sequences (torch.Tensor): Processed radar sequences
        - vibration_sequences (torch.Tensor): Processed vibration sequences
        - labels (torch.Tensor): Corresponding labels
    """
    # Load raw data
    radar_df = pd.read_csv(radar_path)
    vibration_df = pd.read_csv(vibration_path)

    # Feature selection and normalization
    radar_features = ["Distance(m)", "Speed(m/s)", "Energy"]
    vibration_features = ["X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"]
    
    scaler = MinMaxScaler()
    radar_df[radar_features] = scaler.fit_transform(radar_df[radar_features])
    vibration_df[vibration_features] = scaler.fit_transform(vibration_df[vibration_features])

    # Align data lengths
    min_length = min(len(radar_df), len(vibration_df))
    labels = radar_df["Label"].apply(lambda x: 1 if x == "Fall" else 0).values[:min_length]

    # Create sliding window sequences
    def create_sequences(data: np.ndarray, labels: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        sequences, targets = [], []
        for i in range(len(data) - seq_length):
            sequences.append(data[i : i + seq_length])
            if i + seq_length < len(labels):
                targets.append(labels[i + seq_length])
        return np.array(sequences), np.array(targets)

    radar_sequences, radar_labels = create_sequences(radar_df[radar_features].values[:min_length], labels, sequence_length)
    vibration_sequences, _ = create_sequences(vibration_df[vibration_features].values[:min_length], labels, sequence_length)

    # Convert to tensors
    return (
        torch.tensor(radar_sequences, dtype=torch.float32),
        torch.tensor(vibration_sequences, dtype=torch.float32),
        torch.tensor(radar_labels, dtype=torch.float32).view(-1, 1),
    )

def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = DEFAULT_EPOCHS,
) -> Dict[str, List[float]]:
    """Training loop for multimodal model.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Model optimizer
        device (torch.device): Computation device
        epochs (int): Number of training epochs
    
    Returns:
        Dictionary containing training metrics
    """
    model.train()
    metrics = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for radar, vibration, labels in train_loader:
            radar, vibration, labels = radar.to(device), vibration.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(radar, vibration)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        metrics["loss"].append(epoch_loss)
        metrics["accuracy"].append(epoch_acc)

        print(
            f"Epoch [{epoch+1:03d}/{epochs}] | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: {epoch_acc:.4f}"
        )
    
    return metrics

def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Evaluate model performance.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device (torch.device): Computation device
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for radar, vibration, labels in test_loader:
            radar, vibration, labels = radar.to(device), vibration.to(device), labels.to(device)
            outputs = model(radar, vibration)
            
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(outputs.cpu().numpy().flatten())

    # Calculate metrics
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    
    return {
        "true_labels": np.array(all_labels),
        "pred_probs": np.array(all_preds),
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(all_labels, preds_binary),
        "classification_report": classification_report(all_labels, preds_binary, output_dict=True),
    }

def plot_metrics(metrics: Dict[str, np.ndarray]) -> None:
    """Visualize evaluation metrics.
    
    Args:
        metrics (Dict): Dictionary containing evaluation metrics
    """
    # ROC Curve
    plt.figure()
    plt.plot(metrics["fpr"], metrics["tpr"], 
             label=f'ROC Curve (AUC = {metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics["confusion_matrix"],
        display_labels=["Non-Fall", "Fall"],
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

def main(
    radar_data_path: str,
    vibration_data_path: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> None:
    """Main execution pipeline.
    
    Args:
        radar_data_path (str): Path to radar data CSV
        vibration_data_path (str): Path to vibration data CSV
        batch_size (int): Training batch size
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
    """
    # Set up device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)

    # Load and preprocess data
    radar, vibration, labels = load_and_preprocess_data(radar_data_path, vibration_data_path)
    dataset = MultimodalDataset(radar, vibration, labels)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = MultimodalCNNLSTM().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training phase
    print("Starting model training...")
    train_metrics = train(model, train_loader, criterion, optimizer, device, epochs)
    
    # Evaluation phase
    print("\nEvaluating model performance...")
    eval_metrics = evaluate(model, test_loader, device)
    
    # Display results
    print("\nClassification Report:")
    print(pd.DataFrame(eval_metrics["classification_report"]).transpose())
    
    plot_metrics(eval_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal Fall Detection System"
    )
    parser.add_argument(
        "--radar",
        type=str,
        required=True,
        help="Path to radar data CSV file",
    )
    parser.add_argument(
        "--vibration",
        type=str,
        required=True,
        help="Path to vibration data CSV file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    
    args = parser.parse_args()
    main(
        radar_data_path=args.radar,
        vibration_data_path=args.vibration,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
    )