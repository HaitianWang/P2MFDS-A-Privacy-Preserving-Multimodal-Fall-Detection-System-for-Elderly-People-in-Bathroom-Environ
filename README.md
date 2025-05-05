# PPMFD-A-Privacy-Preserving-Multimodal-Fall-Detection-Network-for-Elderly-Individuals-in-Bathroom

PPMFD (Privacy-Preserving Multimodal Fall Detection) is a novel and practical AI-based system designed to monitor and detect fall events in residential bathroom environments, especially for elderly individuals. It leverages multimodal sensors—millimeter-wave radar and triaxial vibration accelerometers—to deliver real-time, privacy-preserving detection with high accuracy.

---

## 📑 Table of Contents

- [1. Introduction](#1-introduction)
- [2. System Overview](#2-system-overview)
  - [2.1 Sensor Configuration](#21-sensor-configuration)
  - [2.2 Deployment Environment](#22-deployment-environment)
- [3. Scenario Simulation and Dataset](#3-scenario-simulation-and-dataset)
- [4. Sensor Evaluation Framework](#4-sensor-evaluation-framework)
  - [4.1 Evaluation Summary](#41-evaluation-summary)
  - [4.2 Scoring Criteria](#42-scoring-criteria)
- [5. Model Architecture](#5-model-architecture)
  - [5.1 Radar Stream](#51-radar-stream)
  - [5.2 Vibration Stream](#52-vibration-stream)
  - [5.3 Feature Fusion](#53-feature-fusion)
- [6. Experimental Results](#6-experimental-results)
  - [6.1 Metrics per Scenario](#61-metrics-per-scenario)
  - [6.2 Confusion Matrix](#62-confusion-matrix)
- [7. Ablation Study](#7-ablation-study)
- [8. Benchmark Comparison](#8-benchmark-comparison)
- [9. Citation](#9-citation)

---

## 1. Introduction

Falls in bathrooms are a major health risk for the elderly. Existing methods using cameras or wearable devices raise privacy or usability issues. PPMFD offers a contactless, privacy-first solution using radar and vibration sensors deployed non-invasively.

---

## 2. System Overview

### 2.1 Sensor Configuration

![sensor setting](docs/Figures/Exp_Setting.png）

- **C4001 mmWave Radar** @ 2.2m height for full-body motion tracking.
- **ADXL345 Vibration Sensor** @ ground level for impact detection.
- **ESP32-C3 MCU** for wireless communication and data logging.

### 2.2 Deployment Environment

- Location: Realistic bathroom space with ceramic surfaces, shower partition, and common obstacles.
- Coverage: Entire activity zone including toilet, floor, and shower corner.

---

## 3. Scenario Simulation and Dataset

<img src="./docs/figures/Exp_behavior.png" width="85%" />

Nine scenarios include:
- (a–c): Environmental triggers (empty, object drops)
- (d–h): Human movement variations
- (i): Simulated falls

> Total: 3+ hours of synchronized multimodal data.

---

## 4. Sensor Evaluation Framework

### 4.1 Evaluation Summary

<img src="./docs/figures/Privacy-Focused Modality Evaluation Summary.png" width="90%" />

> mmWave + Vibration = highest usability for privacy-preserving fall detection.

### 4.2 Scoring Criteria

<img src="./docs/figures/Scoring Framework for Privacy-Preserving Sensor Selection.png" width="60%" />

Weighted evaluation across 8 dimensions (privacy, energy, cost, etc.).

---

## 5. Model Architecture

### 5.1 Radar Stream

- 1D CNN → BiLSTM → Attention
- Extracts global motion patterns from 3D point clouds

### 5.2 Vibration Stream

- Multi-Scale CNN → SEBlock → Self-Attention
- Captures local impacts and subtle signal shifts

### 5.3 Feature Fusion

<img src="./docs/figures/PPMFD Network.png" width="100%" />

Concatenated embeddings are passed to a detection head for binary classification.

---

## 6. Experimental Results

### 6.1 Metrics per Scenario

<img src="./docs/figures/Scenario-Based PPMFD Performance Metrics.png" width="90%" />

> Overall accuracy: **95.0%** | F1: **91.3%**

### 6.2 Confusion Matrix

<img src="./docs/figures/confusion matrix.png" width="85%" />

> High separation between fall and non-fall events, even under similar motion patterns.

---

## 7. Ablation Study

<img src="./docs/figures/Ablation Study of Multimodal Model Components.png" width="95%" />

Tested various combinations of model blocks (CNN, LSTM, Attention) with and without vibration modality. Best performance with full configuration.

---

## 8. Benchmark Comparison

<img src="./docs/figures/Benchmark Comparison of Fall Detection Methods.png" width="100%" />

Compared against 15 state-of-the-art fall detection models:
- Best **Precision** (94.6%) and top-tier **Recall** (87.8%) in non-vision methods.

---

## 9. Citation

```bibtex
@inproceedings{wang2025ppmfd,
  title={PPMFD: A Privacy-Preserving Multimodal Fall Detection Network for Elderly Individuals in Bathroom},
  author={Haitian Wang and Yiren Wang and Yumeng Miao and Yuliang Zhang and Xinyu Wang and Atif Mansoor},
  booktitle={IEEE Conference on ...},
  year={2025}
}
