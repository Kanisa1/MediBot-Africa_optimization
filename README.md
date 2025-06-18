
# 🧠 MediBot Africa: Predicting Malaria, Typhoid, and Pneumonia from Symptoms

## 🎯 Objective
This project explores the implementation of machine learning models with optimization techniques such as regularization, dropout, early stopping, and learning rate adjustments. The goal is to build accurate models that support early diagnosis of common diseases based on symptoms — improving convergence speed, efficiency, and generalization performance.

---

## 📌 Problem Statement
Millions in underserved African regions suffer from preventable diseases due to late diagnoses. This model supports early detection of **Malaria**, **Typhoid**, and **Pneumonia** using 15 binary symptom indicators, such as fever, headache, and cough.

---

## 📊 Dataset

## Source: [https://www.kaggle.com/datasets/miltonmacgyver/symptom-based-disease-prediction-dataset/data](https://www.kaggle.com/datasets/miltonmacgyver/symptom-based-disease-prediction-dataset/data)

| Description     | Value                                |
|------------------|--------------------------------------|
| Source           | Kaggle              |
| Total Samples    | 4,998                                |
| Features         | 15 binary symptoms                   |
| Target Classes   | 3 (Malaria, Typhoid, Pneumonia)      |

---

## 🧪 Models Implemented

| Model                     | Description                                                   |
|---------------------------|---------------------------------------------------------------|
| **Instance 1 (Baseline)** | Neural Network with no optimization                           |
| **Instance 2**            | RMSprop + Dropout + EarlyStopping                             |
| **Instance 3**            | SGD + L2 Regularization                                       |
| **Instance 4**            | Adam + Dropout + L2 + EarlyStopping                           |
| **Instance 5**            | Logistic Regression (Non-Neural Network, with tuned params)   |

All 5 trained models are saved in the `/saved_models` directory.

---

## 🧠 Model Architecture

The neural network architecture used includes:
- **Input Layer**: 15 features
- **Hidden Layers**: 3 fully connected layers (64, 32, 16 neurons) with `relu` activation
- **Dropout**: 0.3 for regularization
- **Output Layer**: 3 neurons (softmax) for multiclass prediction

📷 See [diagrams/model_architecture.png](./diagrams/model_architecture.png)

---

## 🧮 Optimization Comparison Table

| Instance | Optimizer     | Regularizer     | LR     | Early Stop | Dropout | Layers | Accuracy | F1 Score | Recall | Precision |
|----------|---------------|-----------------|--------|------------|---------|--------|----------|----------|--------|-----------|
| 1        | Default (Adam)| None            | -      | ❌ No       | ❌ No    | 3      | 0.8760   | 0.8760   | 0.8760 | 0.8760    |
| 2        | RMSprop       | None            | 0.001  | ✅ Yes      | ✅ 0.3   | 3      | 0.8973   | 0.8972   | 0.8973 | 0.8974    |
| 3        | SGD           | L2 (0.01)       | 0.001  | ❌ No       | ❌ No    | 3      | 0.8680   | 0.8674   | 0.8680 | 0.8691    |
| 4        | Adam          | L2 + Dropout    | 0.0005 | ✅ Yes      | ✅ 0.3   | 3      | 0.8933   | 0.8931   | 0.8933 | 0.8935    |
| 5        | -             | -               | -      | -          | -       | -      | 0.8893   | 0.8890   | 0.8893 | 0.8892    |

> ✅ *Instance 5 refers to the Logistic Regression model (non-NN), tuned with max_iter=200, C=1.0, solver='liblinear'*

---

## 📈 Discussion of Results

Each training instance was designed to test a unique combination of optimization techniques:

- **Instance 1** served as the baseline, using no optimization — it reached a moderate accuracy but overfit quickly.
- **Instance 2** delivered the highest performance among NN models by combining RMSprop, dropout, and early stopping.
- **Instance 3** showed slower convergence and weaker results due to limited optimization, despite using L2 regularization.
- **Instance 4** performed nearly as well as Instance 2 by combining Adam, L2, dropout, and a lower learning rate for stability.
- **Instance 5 (Logistic Regression)** achieved 88.93% accuracy — strong performance for a simpler model, validating its suitability for clean binary datasets.

### ✅ Best Neural Network:
- **Instance 2 (RMSprop + Dropout + EarlyStopping)** with **0.8973 accuracy**

### 🥇 Best Overall Model:
- **Logistic Regression** (Instance 5) with **0.8893 accuracy**

### 🔍 Takeaway:
- Neural networks perform best when optimized with **multiple techniques**.
- Classical models like **Logistic Regression** remain strong baselines, especially for structured, clean data.

---

## 🔬 Error Analysis

The model was evaluated using the following metrics on the test set:

- ✅ **Accuracy**
- ✅ **Precision**
- ✅ **Recall**
- ✅ **F1 Score**
- ✅ **Confusion Matrix**
- ✅ **Per-Class Evaluation (Bar Plot)**

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, preds, target_names=le.classes_))
````

📊 Per-class scores were visualized to assess class-specific weaknesses.

---

## 💾 How to Run This Project

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/MediBot-Africa.git
   cd MediBot-Africa
   ```

2. Open `notebook.ipynb` in Jupyter or Colab and run all cells.

3. Load the best model:

   ```python
   from tensorflow.keras.models import load_model
   model = load_model("saved_models/model_2_rmsprop.keras")
   ```

---

## 🗂️ Project Structure

```
MediBot-Africa/
├── notebook.ipynb
├── README.md
├── saved_models/
│   ├── model_1_baseline.keras
│   ├── model_2_rmsprop.keras
│   ├── model_3_sgd_l2.keras
│   ├── model_4_adam_combo.keras
│   └── model_5_logistic_regression.pkl
├── diagrams/
│   └── model_architecture.png
```

---

## 🎥 Video Presentation

🎬 [Link to 5-Minute Presentation Video](https://your-video-link.com)

* Covers: Dataset overview, model architectures, optimization techniques, result analysis, and final conclusions.

---

## ✅ Submission Instructions

* 🔗 Submit **GitHub repo link** on Canvas (**Attempt 1**)
* 📦 Submit **Zipped folder of repo** on Canvas (**Attempt 2**)

---

## 🙌 Author

Prepared by **Kanisa Rebecca Majok Thiak**
BSc Software Engineering, ALU Kigali
Machine Learning Optimization Project – 2025

```

---


