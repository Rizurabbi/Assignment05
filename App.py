import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Page config and styling
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.markdown("""
    <style>
        .big-font {
            font-size: 20px !important;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            color: gray;
            font-size: small;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    
    st.markdown("### 👨‍💻 Developed by")
    st.markdown("**Rizuanul Alam**  \n📧 rizuanrabbi@email.com  \n🌐 [GitHub](https://github.com/rizurabbi)")
    st.markdown("---")
    st.markdown("📩 Feel free to reach out if you have any suggestions!")


st.title("💳 Credit Card Fraud Detection App")
st.markdown("Welcome! This tool helps you train, evaluate, and predict credit card fraud using machine learning. 🚀")

# Upload training data
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📁 Preview of Uploaded Data")
    st.write(df.head())

    df = df.dropna()
    df = df.sample(frac=1, random_state=42)
    df = df.iloc[:84000]

    fraud = df[df['Class'] == 1]
    legit = df[df['Class'] == 0].sample(len(fraud) * 4, random_state=42)
    balanced_df = pd.concat([fraud, legit]).sample(frac=1, random_state=42)

    X = balanced_df.drop(['Class'], axis=1)
    y = balanced_df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    st.subheader("⚙️ Training Decision Tree...")
    dtree = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    params = {'max_depth': [4, 6, 8], 'min_samples_split': [2, 5]}
    grid = GridSearchCV(dtree, params, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1 = report['1']['f1-score']

    st.subheader("📊 Model Performance on Test Set")
    st.write(f"**AUC Score:** {auc:.4f}")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")
    st.success("✅ Model training and evaluation completed successfully.")

    # Predict on new data
    st.subheader("🔍 Upload Data to Predict Fraud Probability")
    predict_file = st.file_uploader("Upload a CSV to Predict Fraud Probabilities", type=["csv"], key="predict")

    if predict_file is not None:
        try:
            pred_df = pd.read_csv(predict_file)
            pred_df_clean = pred_df.dropna()
            expected_features = X.columns.tolist()
            X_pred = pred_df_clean[expected_features]
            X_pred_scaled = scaler.transform(X_pred)
            fraud_probs = best_model.predict_proba(X_pred_scaled)[:, 1]
            predictions = best_model.predict(X_pred_scaled)

            pred_df_clean['Fraud_Probability'] = fraud_probs
            pred_df_clean['Prediction'] = predictions

            st.write("📋 Predictions with Fraud Probability:")
            st.dataframe(pred_df_clean.head(20))

            fraud_percent = (pred_df_clean['Prediction'] == 1).mean() * 100
            nonfraud_percent = 100 - fraud_percent

            st.subheader("📈 Final Prediction Summary")
            st.write(f"**Fraudulent Transactions:** {fraud_percent:.2f}%")
            st.write(f"**Non-Fraudulent Transactions:** {nonfraud_percent:.2f}%")

            fig, ax = plt.subplots()
            labels = ['Non-Fraud', 'Fraud']
            sizes = [nonfraud_percent, fraud_percent]
            colors = ["#9A76A0", "#a31512"]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')
            st.pyplot(fig)

            csv = pred_df_clean.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", csv, "fraud_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")

# Footer
st.markdown('<div class="footer">Made by Rizuanul Alam | © 2025</div>', unsafe_allow_html=True)
