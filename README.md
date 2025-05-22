# Fraud Detection Web App

This is a web application that uses a machine learning model to detect fraudulent transactions. Built using **Streamlit**, it takes user input in CSV format and returns predictions on whether transactions are fraudulent or not.

## 🚀 Live App

[Click here to view the live app](https://fraud-detection-rizuan.streamlit.app/)


## 🧠 How It Works

1. Upload a `.csv` file with the correct structure.
2. The model will process the data and classify each transaction as **Fraud** or **Not Fraud**.
3. Results are displayed interactively.

## 🧰 Tech Stack

- Python 3.11+
- Streamlit
- pandas
- scikit-learn >= 1.2.0

## 🔧 Setup Instructions

### 🔗 Run locally

```bash
# Clone the repository
git clone https://github.com/your-username/assignment05.git
cd assignment05

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run App.py


