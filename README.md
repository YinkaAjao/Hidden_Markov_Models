# Human Activity Recognition Using Hidden Markov Models

A machine learning pipeline that uses Hidden Markov Models (HMM) to classify human physical activities from raw smartphone accelerometer and gyroscope data. This project serves as a foundational layer for automated athletic performance assessment, supporting the development of a scalable, data-driven platform for global grassroots scouting and talent discovery.

---

## Project Overview

This repository implements an end-to-end unsupervised learning pipeline to capture the probabilistic transitions between distinct physical states: **Standing**, **Walking**, **Jumping**, and **Still**. By translating continuous wearable sensor streams into discrete, recognizable movement states, the system objectively quantifies movement efficiency and physical exertion without the need for manual video tagging.

## Key Features

*   **Sensor Harmonization:** Utilizes pure NumPy linear interpolation to synchronize asynchronous nanosecond hardware logs and rigidly enforce a 100Hz timeline, eliminating baseline drift and memory overflow errors.
*   **Feature Engineering:** Extracts critical time-domain features (Mean, RMS, Variance, Signal Magnitude Area, Inter-axis Correlation) and frequency-domain features (Dominant Frequency, Spectral Energy via FFT) from 1-second sliding windows.
*   **Z-Score Standardization:** Normalizes the feature matrix to ensure high-magnitude frequency metrics do not mathematically destabilize the Gaussian clustering logic.
*   **Unsupervised Training:** Employs the Baum-Welch algorithm (Expectation-Maximization) to train transition and emission probabilities, using a strict log-likelihood convergence criterion (Delta log L < 10^-4).
*   **Sequence Decoding:** Leverages the Viterbi algorithm to map latent internal states to physical ground-truth labels based on kinetic intensity, decoding the most likely sequence of unseen activities.

---

## Repository Structure

*   `data/`: Directory containing the raw, chronological CSV sensor logs used for model training.
*   `test_data/`: Directory containing the isolated, unseen recording sessions used exclusively for model evaluation.
*   `Hidden_Markov_Models.ipynb`: The primary Jupyter Notebook containing the end-to-end data processing, feature extraction, and model implementation pipeline.
*   `hmm_features_checkpoint.npz`: Compressed local archive containing the unscaled feature matrix, numeric state labels, and sequence boundaries.

---

## Installation and Setup

**1. Clone the repository**
git clone https://github.com/YinkaAjao/Hidden_Markov_Models.git 

**2. Install required dependencies**
pip install numpy pandas scipy scikit-learn hmmlearn matplotlib seaborn

**3. Run the pipeline**
Open the Jupyter Notebook to execute the data harmonization, train the Gaussian HMM, and visualize the transition/emission matrices:

jupyter notebook Hidden_Markov_Models.ipynb


## Evaluation and Results
The trained model was tested against an entirely unseen dataset of 20 independent recording sessions. Unsupervised internal states were dynamically aligned to physical labels by analyzing the Z-scores of the variance features (Still < Standing < Walking < Jumping).

**Performance on Unseen Data:**

Still: 1.000 Sensitivity | 0.841 Specificity

Walking: 0.820 Sensitivity | 0.838 Specificity

Standing: 0.548 Sensitivity | 1.000 Specificity

Jumping: 0.527 Sensitivity | 0.946 Specificity

Overall Sequence Accuracy: 71.5%

The Markov Transition Probability Matrix confirmed extremely high self-transition persistence (e.g., > 0.95 along the diagonal), mathematically reflecting realistic human biomechanics where physical states persist over time rather than erratically switching frame-by-frame.

**Author**
David Olayinka Ajao
Software Engineer & Data Science Enthusiast
