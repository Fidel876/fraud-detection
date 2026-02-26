# 🔍 Fraud Detection AI System

An AI-powered system for detecting fraudulent financial transactions using unsupervised anomaly detection. Built with **Isolation Forest**, it can identify suspicious activity in highly imbalanced datasets — no labeled fraud examples required at training time.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model](#model)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Credit card fraud causes billions in losses annually. Traditional rule-based systems struggle with evolving fraud patterns. This project tackles the problem using **Isolation Forest**, an unsupervised ML algorithm that isolates anomalies by randomly partitioning the feature space — anomalous transactions are isolated much faster than normal ones.

The system includes:
- A trained model pipeline that handles class imbalance
- A real-time web interface for transaction analysis
- Evaluation scripts with standard fraud detection metrics

---

## Features

- ✅ Detects anomalous/fraudulent transactions in real time
- ✅ Handles highly imbalanced datasets (fraud is typically <1% of transactions)
- ✅ Interactive web UI built with Streamlit
- ✅ Unsupervised approach — no labelled fraud data needed during training
- ✅ Easily extensible to other anomaly detection models
- ✅ Dev container support for instant cloud development (GitHub Codespaces)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| ML Framework | Scikit-learn |
| Algorithm | Isolation Forest |
| Web Interface | Streamlit |
| Model Persistence | joblib / pickle |
| Environment | `.devcontainer` (Codespaces-ready) |

---

## Project Structure

```
Fraud-detection-ai/
│
├── .devcontainer/          # Dev container config for GitHub Codespaces
├── .streamlit/             # Streamlit configuration (theme, server settings)
├── model/                  # Saved/serialized trained model artifacts
│
├── train_model.py          # Script to train the Isolation Forest model
├── evaluate_model.py       # Script to evaluate model performance
├── app.py                  # Streamlit web application (main entry point)
│
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python runtime version specification
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

---

## Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle.

- 📦 **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions, of which only **492 are fraudulent** (~0.17%)
- Features are **PCA-transformed** (V1–V28) for confidentiality, plus `Time`, `Amount`, and `Class`
- `Class = 1` → Fraud, `Class = 0` → Legitimate

### Setup

1. Download `creditcard.csv` from Kaggle
2. Place it in the project root directory (it is listed in `.gitignore` and not committed)

---

## Model

### Isolation Forest

Isolation Forest is an unsupervised anomaly detection algorithm that works by:

1. Randomly selecting a feature and then randomly selecting a split value between the feature's min and max
2. Recursively partitioning data into isolation trees
3. Anomalies (fraudulent transactions) require **fewer splits** to be isolated, resulting in shorter path lengths

**Key hyperparameters:**
| Parameter | Description |
|---|---|
| `contamination` | Expected proportion of outliers in the dataset |
| `n_estimators` | Number of isolation trees |
| `max_samples` | Number of samples to draw for each tree |
| `random_state` | Seed for reproducibility |

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Fidel876/Fraud-detection-ai.git
cd Fraud-detection-ai

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python train_model.py
```

This will train the Isolation Forest on the dataset and save the model artifact to the `model/` directory.

### Launch the Web App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage

Once the Streamlit app is running, you can:

1. **Upload a CSV file** of transactions to batch-score them for fraud
2. **Enter transaction details manually** via input fields for single-transaction analysis
3. **View results** — each transaction is flagged as `Fraudulent` or `Legitimate` with an anomaly score
4. **Explore visualizations** of the anomaly score distribution

---

## Evaluation

Run the evaluation script to assess model performance on the dataset:

```bash
python evaluate_model.py
```

Metrics reported include:

| Metric | Description |
|---|---|
| Precision | Of flagged transactions, how many are actually fraud |
| Recall | Of all actual fraud, how many were caught |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Area under the ROC curve |
| Confusion Matrix | True/False Positives and Negatives |

> **Note:** Because the dataset is extremely imbalanced, accuracy alone is a misleading metric. Recall and F1 are the most meaningful indicators for fraud detection.

---

## Development with GitHub Codespaces

This repo includes a `.devcontainer` configuration. To spin up a full dev environment instantly:

1. Click the green **Code** button on the GitHub repo page
2. Select **Open with Codespaces → New Codespace**
3. All dependencies will be installed automatically

---

## Contributing

Contributions are welcome! Here's how to get involved:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

### Ideas for Contribution
- Add support for additional models (XGBoost, Autoencoders, LOF)
- Integrate SHAP for model explainability
- Add real-time alerting or email notifications
- Improve UI with charts and transaction history
- Add Docker support for containerized deployment

---

## License

This project is open source. See the repository for licensing details.

---

## Acknowledgements

- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — MLG, ULB
- [Scikit-learn Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Streamlit](https://streamlit.io/) for making ML web apps simple

---

*Built with ❤️ by [Fidel876](https://github.com/Fidel876)*
