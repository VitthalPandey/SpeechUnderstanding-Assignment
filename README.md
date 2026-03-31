# Speech Understanding Assignment

**Vitthal Pandey (M25DE1060)**

---

# 📌 Overview

This assignment covers three major components of speech understanding:

* **Q1:** Cepstral Feature Extraction & Phoneme Boundary Detection
* **Q2:** Paper Implementation – Environment-agnostic Speaker Recognition
* **Q3:** Ethical Auditing & Privacy-Preserving Speech Processing

All experiments are performed using real-world speech datasets without synthetic data generation.

---

# 📂 Project Structure

```
Assignment/
│
├── Q1/
│   ├── mfcc_manual.py
│   ├── leakage_snr.py
│   ├── voiced_unvoiced.py
│   ├── phonetic_mapping.py
│   ├── data/
│   ├── manifest.txt
│   └── q1_report.pdf
│
├── Q2/
│   ├── train.py
│   ├── eval.py
│   ├── configs/
│   ├── results/
│   ├── review.pdf
│   └── q2_readme.md
│
├── Q3/
│   ├── audit.py
│   ├── privacymodule.py
│   ├── train_fair.py
│   ├── evaluation_scripts/
│   ├── examples/
│   └── q3_report.pdf
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation & Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

# 📦 Requirements

```
numpy
scipy
matplotlib
soundfile
torch
torchaudio
transformers
datasets
pandas
```

---

# 📥 Dataset

We use the **LibriSpeech (train-clean-100 subset)** for all experiments.

* Real human speech recordings
* No synthetic or generated data used

---

# ▶️ Question 1: Cepstral Analysis & Boundary Detection

## 🔹 Description

Implements a full speech processing pipeline from scratch, including MFCC extraction, spectral analysis, and phoneme segmentation.

---

## 🔹 How to Run

```bash
cd Q1

python mfcc_manual.py
python leakage_snr.py
python voiced_unvoiced.py
python phonetic_mapping.py
```

---

## 🔹 Components

### MFCC Extraction

* Pre-emphasis
* Framing + windowing
* FFT
* Mel filterbank
* Log compression
* DCT

---

### Spectral Leakage Analysis

* Rectangular vs Hamming vs Hanning windows
* Comparison plots

---

### Voiced/Unvoiced Detection

* Cepstral analysis
* Energy-based segmentation

---

### Phonetic Mapping

Uses **Wav2Vec2** for transcription and alignment.


## 🔹 Outputs

* MFCC heatmap
* Leakage comparison plots
* Voiced/unvoiced segmentation
* RMSE table

---

# ▶️ Question 2: Paper Implementation

## 🔹 Paper

**Disentangled Representation Learning for Environment-agnostic Speaker Recognition**

---

## 🔹 Objectives

* Understand and critique the proposed method
* Implement a simplified version of the model
* Compare with a baseline speaker recognition system

---

## 🔹 Implementation Details

### Model

* Feature extractor (MFCC / embeddings)
* Disentangled representation learning
* Speaker classification head

---

### Baseline

* Standard speaker embedding model
* Compared using classification accuracy

---

### Evaluation Metrics

* Accuracy
* Loss curves
* Confusion matrix

---

## 🔹 How to Run

```bash
cd Q2

python train.py
python eval.py
```

---

## 🔹 Deliverables

* `review.pdf` → critical analysis of paper
* `results/` → plots and tables
* `configs/` → model settings

---

## 🔹 Proposed Improvement

A lightweight regularization technique was introduced to improve disentanglement and robustness under noisy environments.

---

# ▶️ Question 3: Ethical Auditing & Fairness

## 🔹 Objective

To analyze bias in speech datasets and design privacy-preserving and fairness-aware models.

---

## 🔹 Components

### Bias Audit

* Analyze dataset for imbalance
* Factors: gender, age, accent

---

### Privacy-Preserving Module

* Modify speech characteristics
* Preserve linguistic content
* Avoid synthetic dataset creation

---

### Fairness Loss

Custom loss function added to reduce performance gaps across groups.

---

### Evaluation

* DNSMOS / proxy metrics
* Audio quality assessment
* Fairness gap comparison

---

## 🔹 How to Run

```bash
cd Q3

python audit.py
python train_fair.py
```

---

## 🔹 Outputs

* Bias distribution plots
* Audio transformation examples
* Fairness evaluation results

---

# ⚠️ Notes

* All experiments use real-world datasets only
* No synthetic audio generation is used
* Code is modular and reproducible

---

# 🚀 Conclusion

This assignment demonstrates:

* End-to-end speech feature extraction
* Advanced representation learning
* Ethical AI practices in speech systems

---

# 👤 Author

Vitthal Pandey
Roll No: M25DE1060
