# P2MFDS: A Privacy-Preserving Multimodal Fall Detection System for Elderly People in Bathroom Environments

> **Notice:** Our manuscript is currently under peer review. In the meantime, we have released all data-processing pipelines, feature-extraction scripts, and the P2MFDS model code for early access and community feedback. We warmly invite reviewers and fellow researchers to explore the repository, experiment with the AI network, and share your insights or issues. The full multimodal fall-detection dataset will be made publicly available upon publication of the paper.


## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Motivation and Background](#2-motivation-and-background)
- [3. Multimodal Sensor Selection](#3-multimodal-sensor-selection)
- [4. System Setup and Dataset Collection](#4-system-setup-and-dataset-collection)
- [5. Model Architecture](#5-model-architecture)
  - [5.1 Radar Stream](#51-radar-stream)
  - [5.2 Vibration Stream](#52-vibration-stream)
  - [5.3 Feature Fusion and Classifier](#53-feature-fusion-and-classifier)
- [6. Experimental Design](#6-experimental-design)
- [7. Results and Performance](#7-results-and-performance)
  - [7.1 Scenario-wise Metrics](#71-scenario-wise-metrics)
  - [7.2 Comparison with State-of-the-Art](#72-comparison-with-state-of-the-art)
  - [7.3 Ablation Study](#73-ablation-study)
- [8. Conclusion and Future Work](#8-conclusion-and-future-work)
- [9. Main Contributors](#9-main-contributors)

---

## 1. Introduction

Falls are among the leading causes of injury-related deaths in the elderly population. As global aging accelerates, with over 16% of people projected to be aged 65+ by 2050, the demand for reliable, real-time fall detection systems becomes increasingly critical—especially in wet and confined spaces like bathrooms, where over 80% of falls occur.

P2MFDS (Privacy-Preserving Multimodal Fall Detection) is a robust AI-based fall detection system that integrates mmWave radar and vibration sensors. The network not only delivers high accuracy but also preserves personal privacy by avoiding cameras or wearable sensors, making it ideal for private home environments.

![Overview Diagram](./docs/Figures/Overview_Diagram.png)

---

## 2. Motivation and Background

Traditional fall detection methods suffer from various limitations. Wearable devices often face poor user compliance and discomfort, especially in wet environments. Camera-based solutions raise significant privacy concerns, particularly in sensitive locations like bathrooms. Audio-based solutions are affected by environmental noise and false positives.

To overcome these challenges, the research focuses on non-intrusive, privacy-aware modalities. However, unimodal systems based on single modality (e.g., radar or vibration only) are inherently limited by environmental conditions, leading to system bias and suboptimal performance. This motivated the need for a multimodal fusion system capable of capturing both motion dynamics and impact signals.

---

## 3. Multimodal Sensor Selection

A comprehensive sensor evaluation framework was developed to select the most suitable combination for fall detection in bathroom settings. 14 different common modalities including Wi-Fi, infrared, thermal imaging, and vibration were assessed based on 7 criteria: target relevance, recall, energy efficiency, deployability, privacy, availability, and cost.

![Sensor Scoring Framework](./docs/Figures/Scoring%20Framework%20for%20Privacy-Preserving%20Sensor%20Selection.png)

The final selection was based on a weighted scoring system. As shown below, mmWave radar and 3D vibration sensors achieved the highest usability scores, demonstrating superior suitability for accurate and privacy-conscious monitoring.

![Privacy-Focused Modality Evaluation Summary](./docs/Figures/Privacy-Focused%20Modality%20Evaluation%20Summary.png)

---

## 4. System Setup and Dataset Collection

The experiment was conducted in a replicated bathroom environment (2.5m × 1.1m × 2.2m), equipped with typical fixtures like tiled walls, a glass partition, anti-slip flooring, and hygiene items. The mmWave radar was mounted at a height of 2.2 meters, while the ADXL345 vibration sensor was installed on the shower floor.

![bathroom_floorplan](./docs/Figures/bathroom_floorplan.png)

![Setup Diagram](./docs/Figures/Exp_Setting.png)

A total of 48 volunteers participated in data collection, simulating 8 types of daily bathroom activities, including both falls and non-fall movements (e.g., walking, squatting, object drops). Over 120,000 vibration samples and 18,000 mmWave point cloud frames were collected, establishing a substantial multimodal dataset for analysis.

We have collected and constructed a large‐scale, multimodal fall‐detection dataset specifically for elderly users in bathroom environments. It comprises 9 representative scenarios—empty room, light/heavy object drops, normal and bent-posture walking, wall-supported walking, static standing, squatting, and intentional falls—with each scenario recorded for 20 minutes (3 hours total). MmWave radar data were sampled at 10 Hz (18 000 point clouds frames collected) and triaxial vibration at 100 Hz (over 120 000 samples collected), enabling robust fusion of macro‐motion and micro‐impact features. **This dataset will be released concurrently with the publication of this paper.**

---

## 5. Experimental Design

Eight real-world scenarios were designed to test system robustness:
1. Empty bathroom
2. Light object drop (0.7kg)
3. Heavy object drop (2kg)
4. Normal walking
5. Bent posture walk
6. Assisted walk (wall-supported)
7. Static standing / squatting
8. Realistic fall

Each activity was performed for approximately 20 minutes. Fall scenarios were simulated under expert supervision. The synchronized multimodal dataset offers a valuable benchmark for future research.

![Scenario Types](./docs/Figures/Exp_behavior.png)

---

## 6. Model Architecture

![P2MFDS Network](./docs/Figures/P2MFDS_Network.png)

P2MFDS is a dual-stream multimodal neural network. It combines:
- A **CNN-BiLSTM-Attention** pipeline to analyze mmWave radar signals
- A **Multi-Scale CNN + SEBlock + Self-Attention** pipeline for vibration data
- A fusion module that integrates both feature representations

### 6.1 Vibration Stream
Vibration data is fed into a Multi-Scale CNN, allowing detection of subtle and high-impact signals. Channel-wise attention (SEBlock) emphasizes dominant signals, and a self-attention mechanism captures temporal dependencies, reducing noise.

### 6.2 Radar Stream
The mmWave radar signal is transformed into a temporal sequence of point clouds. These are processed via a 1D CNN to extract local motion features, followed by a BiLSTM layer that models long-range dependencies. Attention weights further enhance informative temporal segments.

![SEBlock architecture](./docs/Figures/SEBlock.png)

### 6.3 Feature Fusion and Classifier

The output of the two streams is concatenated and passed through fully connected layers for final binary classification (fall vs. non-fall). This architecture enables the model to learn both macro-motion and micro-impact patterns.

---

## 7. Results and Performance

P2MFDS achieved superior performance across all evaluation metrics.

### 7.1 Scenario-wise Metrics

The model maintained high precision and recall in all scenarios. Especially in hard cases like object drops and squatting, the dual-sensor fusion reduced false positives effectively.
![Confusion Matrix](./docs/Figures/confusion%20matrix.png)

![Scenario Metrics](./docs/Figures/Scenario-Based%20P2MFDS%20Performance%20Metrics.png)


### 7.2 Comparison with State-of-the-Art

P2MFDS was benchmarked against 16 leading models. Despite similar or slightly lower recall, it outperformed in precision and overall F1-score—an essential trait for reducing false alarms in real environments.

![Benchmark Comparison](./docs/Figures/Benchmark%20Comparison%20of%20Fall%20Detection%20Methods.png)

### 7.3 Ablation Study

To validate the effectiveness of each component, an ablation study was performed. Results demonstrate that removing attention or one sensor modality significantly reduces performance. The complete model achieved an accuracy of 95.0% and F1-score of 91.1%.

![Ablation Study](./docs/Figures/Ablation%20Study%20of%20Multimodal%20Model%20Components.png)

---

## 8. Conclusion and Future Work

P2MFDS provides a privacy-aware, multimodal approach to real-time fall detection in bathrooms. Its dual-stream architecture captures both large-scale movement and localized impacts, ensuring high accuracy in complex, noisy environments.

In the future, we aim to:
- Expand testing to real elderly subjects under long-term deployment
- Develop adaptive thresholding and noise calibration
- Explore lighter edge versions for real-time in-home integration

The dataset and model will be released to promote open research and reproducibility.

## 9. Main Contributors

This work was conducted by a research team from the University of Western Australia (UWA).

### Lead Contributors
- **Haitian Wang** – Lead researcher and project coordinator. Designed the sensor selection framework and develop overall system and its integration. Conducted extensive human experimentation, collect the raw data and developed the multimodal dataset.
- **Yiren Wang** – Design and led the implementation of the CNN-BiLSTM-Attention pipeline. Conducted the model training and testing. Finalize the evaluation and result visualization.
- **Xinyu Wang** - Assist in completing system and experiments. Participate in the hardware deployment and neural network design work. Assist in Paper writing and submittion.

---

> Contact: haitian.wang@uwa.edu.au  
> Project by: Department of Computer Science and Software Engineering, University of Western Australia, Australia.
