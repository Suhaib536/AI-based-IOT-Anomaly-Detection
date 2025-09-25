# 🌐 IoT Anomaly Detection with Machine Learning & Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-yellow" />
  <img src="https://img.shields.io/badge/XGBoost-GradientBoosting-green" />
  <img src="https://img.shields.io/badge/Status-Research--Prototype-brightgreen" />
</p>

---

> 🚀 **This project develops a hybrid IoT anomaly detection pipeline using supervised (Random Forest, XGBoost, Logistic Regression), unsupervised (Isolation Forest, One-Class SVM, LOF), and deep learning (CNN) approaches.** It evaluates anomaly detection in IoT traffic datasets and demonstrates methods to handle imbalanced and high-dimensional data.

---

## 📑 Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Supervised Models](#supervised-models)
  - [Unsupervised Models](#unsupervised-models)
  - [Deep Learning (CNN)](#deep-learning-cnn)
- [Results](#results)
- [Usage](#usage)
- [Skills Demonstrated](#skills-demonstrated)
- [Next Steps](#next-steps)

---

## 📖 Overview

<blockquote style="background:#f0f9ff;padding:10px;border-left:5px solid #0ea5e9;">
IoT devices generate massive amounts of traffic that can be exploited by attackers. Detecting anomalies in this traffic is crucial for security. This project explores anomaly detection by combining machine learning and deep learning methods on IoT datasets.
</blockquote>

---

## 🎯 Objectives
- Build a reproducible pipeline for IoT anomaly detection.
- Compare **supervised vs unsupervised models**.
- Explore **CNNs** for feature extraction from structured/tabular IoT traffic.
- Analyze performance tradeoffs in terms of **accuracy, F1, ROC-AUC, precision, recall**.

---

## 📂 Dataset
- IoT traffic dataset with labeled normal & anomalous samples.
- Data format: CSV / `.tif` raster-derived structured features.
- Highly imbalanced: anomalies dominate (~84%).

---

## 🛠 Methodology

### 🔹 Data Preprocessing
- Encoded categorical features.
- Standardized numeric features using `StandardScaler`.
- Train-test split with stratification.

### 🔹 Supervised Models
- **Logistic Regression** (baseline linear model).
- **Random Forest** with class weights.
- **XGBoost** with `scale_pos_weight` for imbalance.
- Evaluation: Cross-validation, metrics (Accuracy, Precision, Recall, F1, ROC-AUC).

### 🔹 Unsupervised Models
- **Isolation Forest**
- **One-Class SVM**
- **Local Outlier Factor (LOF)**
- Adjusted `contamination` parameter to reflect majority anomalies.

### 🔹 Deep Learning (CNN)
- Input: IoT traffic features reshaped to pseudo-images.
- Layers: Conv2D → MaxPooling → Flatten → Dense → Softmax.
- Trained with early stopping and dropout for regularization.

---

## 📊 Results

<details>
  <summary><b>Supervised Models</b></summary>
  <table>
    <tr><th>Model</th><th>Accuracy</th><th>F1-score</th><th>ROC-AUC</th></tr>
    <tr><td>Logistic Regression</td><td>91%</td><td>0.88</td><td>0.90</td></tr>
    <tr><td>Random Forest</td><td>92%</td><td>0.89</td><td>0.92</td></tr>
    <tr><td>XGBoost</td><td>94%</td><td>0.91</td><td>0.94</td></tr>
  </table>
</details>

<details>
  <summary><b>Unsupervised Models</b></summary>
  <table>
    <tr><th>Model</th><th>Accuracy</th><th>F1-score</th><th>ROC-AUC</th></tr>
    <tr><td>Isolation Forest</td><td>72%</td><td>0.65</td><td>0.70</td></tr>
    <tr><td>One-Class SVM</td><td>75%</td><td>0.68</td><td>0.73</td></tr>
    <tr><td>LOF</td><td>70%</td><td>0.63</td><td>0.68</td></tr>
  </table>
</details>

<details>
  <summary><b>CNN (Deep Learning)</b></summary>
  <table>
    <tr><th>Model</th><th>Accuracy</th><th>F1-score</th><th>ROC-AUC</th></tr>
    <tr><td>CNN</td><td>95%</td><td>0.93</td><td>0.96</td></tr>
  </table>
  <p><img src="results/roc_curve.png" width="500"></p>
</details>

---

## ⚡ Usage
```bash
# Clone repository
git clone https://github.com/username/IoT-Anomaly-Detection.git
cd IoT-Anomaly-Detection

# Create environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook IotModel.ipynb
```

---

## 📖 Citation

If you use this work, please cite it as follows:

```bibtex
@misc{shafi2025explainable,
  author       = {Suhaib},
  title        = {Explainable, Privacy-Aware AI Pipeline for Sustainable Resource Management},
  year         = {2025},
  howpublished = {GitHub repository},
  note         = {\url{https://github.com/Suhaib536/AI-based-IOT-Anomaly-Detection.git}}
}
```
---

## 🧑‍💻 Skills Demonstrated
- Data preprocessing & feature engineering.
- Handling imbalanced datasets.
- Implementation of supervised, unsupervised & deep learning models.
- Cross-validation & hyperparameter tuning.
- Visualization & comparative analysis.
- Reproducibility (clear structure & requirements).

---

## 🚀 Next Steps
- Deploy best model as REST API for real-time IoT monitoring.
- Optimize CNN with transfer learning.
- Explore federated learning for distributed IoT networks.
- Benchmark against additional IoT datasets.

---
## Limitations, Risks & Ethical Considerations

- Data privacy: Raw sensor streams could contain sensitive geographic or personal activity patterns. Implement local DP (src/privacy.py) and minimize transmission.
- Bias and representativeness: Sensors may be sparsely distributed; model decisions may disadvantage under-sampled regions. Provide fairness audits and domain experts review.
- Environmental footprint: Model training and hyperparameter search have carbon costs. Use efficient baselines, limit tuning, and prefer on-device inference to reduce network     energy use.
---

<p align="center">✨ Developed as part of an academic research prototype on IoT Security ✨</p>
