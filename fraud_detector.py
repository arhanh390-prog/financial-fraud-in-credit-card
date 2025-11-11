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

# 1. IMPORT OF IMPORTANT LIB (Done above)

@st.cache_data
def load_data(uploaded_file):
    """Reads the uploaded CSV file into a DataFrame."""
    try:
        data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

@st.cache_resource
def train_model(data, feature_cols, target_col):
    """
    Trains a Logistic Regression model on the data.
    Returns the trained model, scaler, and evaluation metrics.
    """
    # 4. TRAIN AND TEST SPLIT (DECIDING X AND Y)
    X = data[feature_cols]
    y = data[target_col]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data (stratify y to keep class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # 5. LOGISTIC REGRESSION MODEL (FIT)
    # Use class_weight='balanced' to handle imbalanced data
    model = LogisticRegression(class_weight='balanced', solver='lbfgs')
    model.fit(X_train, y_train)

    # 5. LOGISTIC REGRESSION MODEL (PREDICT)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # 6. GENERATE CLASSIFICATION REPORT
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, accuracy, report, cm, feature_cols

def main():
    st.title("🛡️ Logistic Regression Fraud Detector")
    st.markdown("Upload your transaction CSV, clean the data, and build a model.")

    # 2. READING THE CSV
    uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is None:
            return

        st.subheader("1. Data Preview")
        st.dataframe(data.head(), use_container_width=True)

        # 3. DATA CLEANING
        st.subheader("2. Data Cleaning")
        st.write("**Missing Values Check:**")
        null_counts = data.isnull().sum()
        st.dataframe(null_counts[null_counts > 0].to_frame(name='Missing Values'), use_container_width=True)
        
        data_cleaned = data.dropna()
        st.write(f"Original data shape: `{data.shape}`")
        st.write(f"Data shape after dropping rows with missing values: `{data_cleaned.shape}`")
        st.success(f"Removed {data.shape[0] - data_cleaned.shape[0]} rows with missing data.")
        
        # 4. TRAIN/TEST SPLIT (Define X and Y)
        st.subheader("3. Feature & Target Selection")
        st.markdown("Select your feature columns (X) and the one target column (y).")
        
        all_cols = data_cleaned.columns.tolist()
        
        # Try to guess the default target (e.g., 'Class', 'Fraud')
        default_target_index = len(all_cols) - 1 # Default to last column
        for i, col in enumerate(all_cols):
            if col.lower() in ['class', 'fraud', 'is_fraud', 'target']:
                default_target_index = i
                break

        target_col = st.selectbox("Select Target Column (y)", all_cols, index=default_target_index)
        
        # Default features are all columns *except* the selected target
        default_features = [col for col in all_cols if col != target_col]
        
        # Filter to only numeric columns for the model
        numeric_cols = data_cleaned.select_dtypes(include=np.number).columns.tolist()
        available_features = [col for col in default_features if col in numeric_cols]
        
        feature_cols = st.multiselect("Select Feature Columns (X)", 
                                      available_features, 
                                      default=available_features)
        
        st.warning("Only numeric columns are available for selection as features.")

        if not feature_cols:
            st.error("Please select at least one feature column.")
            return
            
        if not target_col:
            st.error("Please select a target column.")
            return

        # 5 & 6. TRAIN MODEL AND GET REPORT
        if st.button("Train Logistic Regression Model", type="primary"):
            st.subheader("4. Model Training & Evaluation")

            # --- NEW VALIDATION BLOCK ---
            # Check if the target column has at least 2 classes
            target_unique_values = data_cleaned[target_col].nunique()
            if target_unique_values < 2:
                st.error(f"**Training Error:** The selected target column ('{target_col}') only has {target_unique_values} unique value.")
                st.warning("A logistic regression model needs at least two different classes in the target column (e.g., 0 and 1) to learn from. This often happens if the data is a summary report.")
                st.info("Please upload a transaction-level dataset or select a different target column that contains both 0s and 1s.")
                return  # Stop execution here
            # --- END OF NEW BLOCK ---

            with st.spinner('Training model... This may take a moment.'):
                try:
                    model, scaler, accuracy, report, cm, trained_features = train_model(data_cleaned, feature_cols, target_col)
                    st.success("Model trained successfully!")

                    # Store model assets in session state for the sidebar
                    st.session_state['model_trained'] = True
                    st.session_state['model_assets'] = (model, scaler, trained_features)

                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Overall Accuracy", f"{accuracy * 100:.2f}%")
                        st.text("Classification Report:")
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                    
                    with col2:
                        st.text("Confusion Matrix:")
                        cm_df = pd.DataFrame(cm,
                                             index=[f"Actual: {i}" for i in model.classes_],
                                             columns=[f"Predicted: {i}" for i in model.classes_])
                        st.dataframe(cm_df, use_container_width=True)
                        
                        tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0,0,0,0)
                        st.markdown(f"""
                        - **True Negatives:** {tn}
                        - **False Positives:** {fp}
                        - **False Negatives:** {fn}
                        - **True Positives:** {tp}
                        """)
                
                except ValueError as ve:
                    st.error(f"Error during model training: {ve}")
                    st.warning("This can happen if your target column is not binary (0 and 1) or if your features have non-numeric data.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    # --- INTERACTIVE PREDICTION SIDEBAR ---
    # Only show if the model has been trained successfully
    if st.session_state.get('model_trained', False):
        st.sidebar.header("🔬 Real-Time Fraud Check")
        st.sidebar.markdown("Enter transaction details to get a prediction.")
        
        model, scaler, trained_features = st.session_state['model_assets']
        
        input_data = {}
        for feature in trained_features:
            input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0, format="%.2f")
            
        if st.sidebar.button("Check Transaction", type="primary"):
            # Create a DataFrame for the input
            input_df = pd.DataFrame([input_data])
            
            # Scale the input data using the *same* scaler
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            # Display the result
            st.sidebar.subheader("Prediction Result")
            class_0, class_1 = model.classes_
            
            if prediction[0] == class_1: # Assuming class 1 is "Fraud"
                prob = prediction_proba[0][1] * 100
                st.sidebar.error(f"Prediction: FRAUD (Class {class_1}) (Confidence: {prob:.2f}%)")
            else:
                prob = prediction_proba[0][0] * 100
                st.sidebar.success(f"Prediction: NOT FRAUD (Class {class_0}) (Confidence: {prob:.2f}%)")

if __name__ == "__main__":
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False
        
    main()
