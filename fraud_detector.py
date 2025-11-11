import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

# Set Streamlit page config
st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_data
def create_mock_data():
    """
    Generates a mock, imbalanced dataset for fraud detection.
    Fraudulent transactions will have different statistical properties.
    """
    # Number of samples
    n_legit = 10000
    n_fraud = 100

    # Generate legitimate transactions
    legit_data = {
        'Amount': np.random.normal(100, 50, n_legit),
        'Time': np.random.randint(0, 172800, n_legit),  # Time in seconds over 2 days
        'V1': np.random.normal(0, 1, n_legit),
        'V2': np.random.normal(0, 1, n_legit),
        'Class': 0
    }
    legit_df = pd.DataFrame(legit_data)
    # Ensure amount is not negative
    legit_df['Amount'] = legit_df['Amount'].apply(lambda x: max(x, 0))

    # Generate fraudulent transactions
    fraud_data = {
        'Amount': np.random.normal(500, 150, n_fraud),  # Frauds have higher amounts
        'Time': np.random.randint(0, 172800, n_fraud),
        'V1': np.random.normal(5, 2, n_fraud),   # Different feature distribution
        'V2': np.random.normal(-5, 2, n_fraud),  # Different feature distribution
        'Class': 1
    }
    fraud_df = pd.DataFrame(fraud_data)
    fraud_df['Amount'] = fraud_df['Amount'].apply(lambda x: max(x, 0))

    # Combine and shuffle
    data = pd.concat([legit_df, fraud_df])
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return data

@st.cache_resource
def train_model(data):
    """
    Trains a Logistic Regression model on the data.
    Returns the trained model and the scaler.
    """
    # Define features (X) and target (y)
    features = ['Amount', 'Time', 'V1', 'V2']
    target = 'Class'
    
    X = data[features]
    y = data[target]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data (stratify y to keep class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train the model
    # Use class_weight='balanced' to handle the imbalanced data
    model = LogisticRegression(class_weight='balanced', solver='lbfgs')
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, X_test, y_test, y_pred, accuracy, report, cm

def main():
    st.title("🛡️ Logistic Regression Fraud Detector")
    st.markdown("A Streamlit app to demonstrate credit card fraud detection using simulated data.")

    # --- 1. LOAD DATA ---
    with st.spinner('Generating simulated data...'):
        data = create_mock_data()

    st.subheader("1. Simulated Transaction Data")
    st.markdown(f"""
    We've generated a mock dataset of **{data.shape[0]}** transactions.
    The data is **highly imbalanced**, which is typical for fraud detection.
    - **Legitimate Transactions (Class 0):** {data['Class'].value_counts()[0]}
    - **Fraudulent Transactions (Class 1):** {data['Class'].value_counts()[1]}
    """)
    st.dataframe(data.head(), use_container_width=True)

    # --- 2. TRAIN MODEL ---
    st.subheader("2. Model Training")
    with st.spinner('Training Logistic Regression model...'):
        model, scaler, X_test, y_test, y_pred, accuracy, report, cm = train_model(data)
        time.sleep(1) # Simulate training time
    
    st.success("Model trained successfully!")
    st.markdown("""
    We've trained a Logistic Regression model. Key steps:
    1.  **Scaled Features:** `Amount`, `Time`, `V1`, and `V2` were scaled using `StandardScaler`.
    2.  **Handled Imbalance:** We used `class_weight='balanced'` in the model to give more importance to the rare fraud class.
    3.  **Split Data:** The data was split 70/30 into training and testing sets.
    """)

    # --- 3. EVALUATE MODEL ---
    st.subheader("3. Model Evaluation")
    st.markdown("Here's how our model performed on the unseen test data.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Overall Accuracy", f"{accuracy * 100:.2f}%")
        st.markdown("Accuracy can be misleading in imbalanced datasets. The metrics below are more important.")
        
        st.text("Classification Report:")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

    with col2:
        st.text("Confusion Matrix:")
        # Displaying confusion matrix in a more readable format
        cm_df = pd.DataFrame(cm,
                             index=[f"Actual: {i}" for i in [0, 1]],
                             columns=[f"Predicted: {i}" for i in [0, 1]])
        st.dataframe(cm_df, use_container_width=True)

        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        - **True Negatives (TN):** {tn} (Correctly identified legit)
        - **False Positives (FP):** {fp} (Incorrectly flagged legit as fraud)
        - **False Negatives (FN):** {fn} (Missed fraud!)
        - **True Positives (TP):** {tp} (Correctly caught fraud)
        """)

    # --- 4. INTERACTIVE PREDICTION ---
    st.sidebar.header("🔬 Real-Time Fraud Check")
    st.sidebar.markdown("Enter transaction details to get a prediction.")

    # Get user input from sidebar
    amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=120.50, step=1.0)
    time_input = st.sidebar.number_input("Time (seconds since first transaction)", min_value=0, value=50000, step=1000)
    v1_input = st.sidebar.slider("V1 (Anonymized Feature)", -10.0, 10.0, 1.5, 0.1)
    v2_input = st.sidebar.slider("V2 (Anonymized Feature)", -10.0, 10.0, -2.0, 0.1)

    if st.sidebar.button("Check Transaction", type="primary"):
        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'Amount': [amount],
            'Time': [time_input],
            'V1': [v1_input],
            'V2': [v2_input]
        })

        # Scale the input data using the *same* scaler
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Display the result in the sidebar
        st.sidebar.subheader("Prediction Result")
        if prediction[0] == 1:
            prob = prediction_proba[0][1] * 100
            st.sidebar.error(f"Prediction: FRAUD (Confidence: {prob:.2f}%)")
        else:
            prob = prediction_proba[0][0] * 100
            st.sidebar.success(f"Prediction: NOT FRAUD (Confidence: {prob:.2f}%)")

if __name__ == "__main__":
    main()
