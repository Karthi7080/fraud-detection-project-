
# 💳 Fraud Detection Project

A machine learning project to detect fraudulent credit card transactions using Isolation Forest and Random Forest classifiers. The app is built and deployed using Streamlit.

---

## 📁 Project Structure

```
fraud-detection-project/
│
├── app.py                  # Streamlit frontend app
├── fraud_detection_model.pkl  # Pretrained Isolation Forest model
├── requirements.txt        # Python package dependencies
└── README.md               # Project overview
```

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Karthi7080/fraud-detection-project.git
cd fraud-detection-project
```

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📊 Features

- Visualizes fraud vs. non-fraud transactions
- Uses Isolation Forest to detect anomalies
- Displays prediction results and fraud count
- Shows confusion matrix and classification report

---

## 📦 Built With

- Python
- Scikit-learn
- Streamlit
- Pandas / NumPy
- Matplotlib / Seaborn
- Joblib

---


## 🙋‍♂️ Author

**Karthik Murali**
