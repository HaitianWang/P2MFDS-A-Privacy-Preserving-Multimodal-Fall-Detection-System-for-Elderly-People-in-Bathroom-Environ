#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fall Detection System using Deep Learning Models

This script implements and compares three deep learning architectures (CNN-LSTM, LSTM, and CNN)
for fall detection using radar data. The models are evaluated using standard metrics including
accuracy, ROC curves, and confusion matrices.
"""

# Standard library imports
import argparse
import os
import sys
from typing import Tuple, Dict, List, Any

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# Constants
SEQUENCE_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Set up matplotlib style for academic papers
plt.style.use("seaborn-poster")
plt.rcParams.update({"font.size": 12, "figure.figsize": (10, 6)})

class FallDataset(Dataset):
    """Custom Dataset class for fall detection data.
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Target labels
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid model for fall detection.
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): LSTM hidden size
        num_layers (int): Number of LSTM layers
        num_classes (int): Number of output classes
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
    ):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = self.dropout(h_n[-1])
        x = self.fc(x)
        return x


class LSTMModel(nn.Module):
    """LSTM model for fall detection.
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): LSTM hidden size
        num_layers (int): Number of LSTM layers
        num_classes (int): Number of output classes
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
    ):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        x = self.dropout(h_n[-1])
        x = self.fc(x)
        return x


class CNNModel(nn.Module):
    """CNN model for fall detection.
    
    Args:
        input_size (int): Number of input features
        num_classes (int): Number of output classes
        sequence_length (int): Length of input sequences
    """
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        sequence_length: int,
    ):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (sequence_length // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences from input data.
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Target labels
        seq_length (int): Sequence length
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Sequences and corresponding labels
    """
    sequences, labels = [], []
    for i in range(len(X) - seq_length):
        sequences.append(X[i : i + seq_length])
        labels.append(y[i + seq_length])
    return np.array(sequences), np.array(labels)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> Tuple[List[float], List[float]]:
    """Train the model and return training metrics.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        epochs (int): Number of training epochs
    
    Returns:
        Tuple[List[float], List[float]]: Training losses and accuracies
    """
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {epoch_loss:.4f}, "
            f"Train Accuracy: {epoch_acc:.4f}"
        )
    
    return train_losses, train_accuracies


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the model on test data.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to evaluate on
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            True labels, predicted labels, and prediction probabilities
    """
    model.eval()
    y_pred_probs, y_pred, y_true = [], [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            y_pred_probs.extend(probabilities)
            
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred), np.array(y_pred_probs)


def plot_metrics(
    train_losses: List[float],
    train_accuracies: List[float],
    model_name: str,
) -> None:
    """Plot training metrics.
    
    Args:
        train_losses (List[float]): Training losses
        train_accuracies (List[float]): Training accuracies
        model_name (str): Name of the model
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(train_losses, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title(f"{model_name} Training Loss")
    
    ax2.plot(train_accuracies, label="Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title(f"{model_name} Training Accuracy")
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    model_name: str,
) -> float:
    """Plot ROC curve and return AUC score.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_probs (np.ndarray): Predicted probabilities
        model_name (str): Name of the model
    
    Returns:
        float: AUC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """Plot confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        model_name (str): Name of the model
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Fall", "Fall"],
        yticklabels=["No Fall", "Fall"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Print classification report and return as DataFrame.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        model_name (str): Name of the model
    
    Returns:
        pd.DataFrame: Classification report as DataFrame
    """
    print(f"\n{classification_report(y_true, y_pred, target_names=['No Fall', 'Fall'])}")
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=["No Fall", "Fall"],
        output_dict=True,
    )
    df_report = pd.DataFrame(report).transpose()
    
    return df_report


def compare_models(
    metrics: Dict[str, Dict[str, Any]],
) -> None:
    """Compare performance of different models.
    
    Args:
        metrics (Dict[str, Dict[str, Any]]): Dictionary containing metrics for each model
    """
    # Plot training loss comparison
    plt.figure(figsize=(8, 5))
    for model_name, model_metrics in metrics.items():
        plt.plot(
            range(1, EPOCHS + 1),
            model_metrics["train_losses"],
            label=f"{model_name} Loss",
            linestyle="-",
            marker="o",
        )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    
    # Plot training accuracy comparison
    plt.figure(figsize=(8, 5))
    for model_name, model_metrics in metrics.items():
        plt.plot(
            range(1, EPOCHS + 1),
            model_metrics["train_accuracies"],
            label=f"{model_name} Accuracy",
            linestyle="-",
            marker="o",
        )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Comparison")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    
    # Plot ROC curve comparison
    plt.figure(figsize=(8, 5))
    for model_name, model_metrics in metrics.items():
        plt.plot(
            model_metrics["fpr"],
            model_metrics["tpr"],
            label=f"{model_name} (AUC = {model_metrics['roc_auc']:.2f})",
        )
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


def main(data_path: str) -> None:
    """Main function to run the fall detection pipeline.
    
    Args:
        data_path (str): Path to the input data file
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    df = pd.read_csv(data_path)
    df["Label"] = df["Label"].map({"Fall": 1, "No Fall": 0})
    
    features = ["Distance(m)", "Speed(m/s)", "Energy"]
    labels = "Label"
    
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    X = df[features].values
    y = df[labels].values
    
    X_seq, y_seq = create_sequences(X, y, SEQUENCE_LENGTH)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq,
        y_seq,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_seq,
    )
    
    # Create DataLoaders
    train_dataset = FallDataset(X_train, y_train)
    test_dataset = FallDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    
    # Dictionary to store metrics for all models
    all_metrics = {}
    
    # CNN-LSTM Model
    print("\nTraining CNN-LSTM Model...")
    cnn_lstm = CNNLSTMModel(
        input_size=len(features),
        hidden_size=128,
        num_layers=2,
        num_classes=2,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_lstm.parameters(), lr=LEARNING_RATE)
    
    cnn_lstm_train_losses, cnn_lstm_train_accuracies = train_model(
        cnn_lstm,
        train_loader,
        criterion,
        optimizer,
        device,
        EPOCHS,
    )
    
    plot_metrics(cnn_lstm_train_losses, cnn_lstm_train_accuracies, "CNN-LSTM")
    
    cnn_lstm_y_true, cnn_lstm_y_pred, cnn_lstm_y_pred_probs = evaluate_model(
        cnn_lstm,
        test_loader,
        device,
    )
    
    cnn_lstm_fpr, cnn_lstm_tpr, _ = roc_curve(
        cnn_lstm_y_true,
        cnn_lstm_y_pred_probs,
    )
    cnn_lstm_roc_auc = auc(cnn_lstm_fpr, cnn_lstm_tpr)
    
    plot_roc_curve(cnn_lstm_y_true, cnn_lstm_y_pred_probs, "CNN-LSTM")
    plot_confusion_matrix(cnn_lstm_y_true, cnn_lstm_y_pred, "CNN-LSTM")
    cnn_lstm_report = print_classification_report(
        cnn_lstm_y_true,
        cnn_lstm_y_pred,
        "CNN-LSTM",
    )
    
    # Save CNN-LSTM model
    torch.save(cnn_lstm.state_dict(), "cnn_lstm_fall_detection.pth")
    print("CNN-LSTM Model saved successfully!")
    
    # Store metrics for comparison
    all_metrics["CNN-LSTM"] = {
        "train_losses": cnn_lstm_train_losses,
        "train_accuracies": cnn_lstm_train_accuracies,
        "fpr": cnn_lstm_fpr,
        "tpr": cnn_lstm_tpr,
        "roc_auc": cnn_lstm_roc_auc,
        "report": cnn_lstm_report,
    }
    
    # LSTM Model
    print("\nTraining LSTM Model...")
    lstm = LSTMModel(
        input_size=len(features),
        hidden_size=128,
        num_layers=2,
        num_classes=2,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm.parameters(), lr=LEARNING_RATE)
    
    lstm_train_losses, lstm_train_accuracies = train_model(
        lstm,
        train_loader,
        criterion,
        optimizer,
        device,
        EPOCHS,
    )
    
    plot_metrics(lstm_train_losses, lstm_train_accuracies, "LSTM")
    
    lstm_y_true, lstm_y_pred, lstm_y_pred_probs = evaluate_model(
        lstm,
        test_loader,
        device,
    )
    
    lstm_fpr, lstm_tpr, _ = roc_curve(lstm_y_true, lstm_y_pred_probs)
    lstm_roc_auc = auc(lstm_fpr, lstm_tpr)
    
    plot_roc_curve(lstm_y_true, lstm_y_pred_probs, "LSTM")
    plot_confusion_matrix(lstm_y_true, lstm_y_pred, "LSTM")
    lstm_report = print_classification_report(lstm_y_true, lstm_y_pred, "LSTM")
    
    # Save LSTM model
    torch.save(lstm.state_dict(), "lstm_fall_detection.pth")
    print("LSTM Model saved successfully!")
    
    # Store metrics for comparison
    all_metrics["LSTM"] = {
        "train_losses": lstm_train_losses,
        "train_accuracies": lstm_train_accuracies,
        "fpr": lstm_fpr,
        "tpr": lstm_tpr,
        "roc_auc": lstm_roc_auc,
        "report": lstm_report,
    }
    
    # CNN Model
    print("\nTraining CNN Model...")
    cnn = CNNModel(
        input_size=len(features),
        num_classes=2,
        sequence_length=SEQUENCE_LENGTH,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    
    cnn_train_losses, cnn_train_accuracies = train_model(
        cnn,
        train_loader,
        criterion,
        optimizer,
        device,
        EPOCHS,
    )
    
    plot_metrics(cnn_train_losses, cnn_train_accuracies, "CNN")
    
    cnn_y_true, cnn_y_pred, cnn_y_pred_probs = evaluate_model(
        cnn,
        test_loader,
        device,
    )
    
    cnn_fpr, cnn_tpr, _ = roc_curve(cnn_y_true, cnn_y_pred_probs)
    cnn_roc_auc = auc(cnn_fpr, cnn_tpr)
    
    plot_roc_curve(cnn_y_true, cnn_y_pred_probs, "CNN")
    plot_confusion_matrix(cnn_y_true, cnn_y_pred, "CNN")
    cnn_report = print_classification_report(cnn_y_true, cnn_y_pred, "CNN")
    
    # Save CNN model
    torch.save(cnn.state_dict(), "cnn_fall_detection.pth")
    print("CNN Model saved successfully!")
    
    # Store metrics for comparison
    all_metrics["CNN"] = {
        "train_losses": cnn_train_losses,
        "train_accuracies": cnn_train_accuracies,
        "fpr": cnn_fpr,
        "tpr": cnn_tpr,
        "roc_auc": cnn_roc_auc,
        "report": cnn_report,
    }
    
    # Compare all models
    compare_models(all_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fall Detection System using Deep Learning Models"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./labeled_radar_data.csv",
        help="Path to the input data file (default: ./labeled_radar_data.csv)",
    )
    args = parser.parse_args()
    
    main(args.data_path)