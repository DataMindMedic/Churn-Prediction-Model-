import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('models/churnguard_best_model.pkl')

# Define the prediction function
def predict_churn(model, input_data):
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    return prediction, probability

def main():
    st.title('ChurnGuard: Customer Churn Prediction')
    st.write('Upload customer data or input values manually to predict churn probability')
    
    # Load the model
    try:
        model = load_model()
        st.success('Model loaded successfully!')
    except Exception as e:
        st.error(f'Error loading model: {e}')
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        st.header("Enter Customer Details")
        
        # Get user input for each feature
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox('Gender', ['Male', 'Female'])
            senior_citizen = st.selectbox('Senior Citizen', [0, 1])
            partner = st.selectbox('Partner', ['Yes', 'No'])
            dependents = st.selectbox('Dependents', ['Yes', 'No'])
            tenure = st.slider('Tenure (months)', 0, 72, 12)
        
        with col2:
            phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
            multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
            internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
            online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
            online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
            tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
            streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
            streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
        
        with col2:
            contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
            paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
            payment_method = st.selectbox('Payment Method', [
                'Electronic check', 
                'Mailed check', 
                'Bank transfer (automatic)', 
                'Credit card (automatic)'
            ])
            monthly_charges = st.slider('Monthly Charges', 0, 150, 65)
            total_charges = st.slider('Total Charges', 0, 10000, 2000)
        
        # Create a dataframe with user inputs
        input_dict = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        input_df = pd.DataFrame([input_dict])
        
        if st.button('Predict Churn'):
            prediction, probability = predict_churn(model, input_df)
            
            # Create a gauge chart for probability
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.barh(['Churn Probability'], [probability], color='red', alpha=0.6)
            ax.barh(['Churn Probability'], [1-probability], left=[probability], color='green', alpha=0.6)
            
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax.set_ylabel('')
            ax.set_title('Churn Probability')
            
            for spine in ax.spines.values():
                spine.set_visible(False)
                
            # Add text annotation for the probability
            ax.text(probability/2, 0, f"{probability*100:.1f}%", 
                    ha='center', va='center', color='white', fontweight='bold')
            ax.text((1+probability)/2, 0, f"{(1-probability)*100:.1f}%", 
                    ha='center', va='center', color='white', fontweight='bold')
                    
            st.pyplot(fig)
            
            # Show prediction
            st.subheader("Prediction Result:")
            if prediction == 1:
                st.error(f"⚠️ Customer is likely to churn (Probability: {probability:.2%})")
                st.write("#### Recommendations to reduce churn risk:")
                st.write("1. Offer a loyalty discount")
                st.write("2. Reach out for feedback on service issues")
                st.write("3. Consider a contract upgrade incentive")
            else:
                st.success(f"✅ Customer is likely to stay (Confidence: {1-probability:.2%})")
                st.write("#### Recommendations to maintain loyalty:")
                st.write("1. Thank them for their continued business")
                st.write("2. Inform about new services they might benefit from")
    
    with tab2:
        st.header("Upload CSV File for Batch Prediction")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.write(data.head())
            
            if st.button('Run Batch Prediction'):
                # Check if customerID exists and save it
                if 'customerID' in data.columns:
                    customer_ids = data['customerID']
                    data = data.drop('customerID', axis=1)
                else:
                    customer_ids = pd.Series([f"Customer_{i}" for i in range(len(data))])
                
                # Check if Churn column exists and remove it
                if 'Churn' in data.columns:
                    data = data.drop('Churn', axis=1)
                
                # Make predictions
                try:
                    predictions = model.predict(data)
                    probabilities = model.predict_proba(data)[:, 1]
                    
                    # Create results dataframe
                    results = pd.DataFrame({
                        'CustomerID': customer_ids,
                        'Churn_Prediction': predictions,
                        'Churn_Probability': probabilities
                    })
                    
                    results['Churn_Prediction'] = results['Churn_Prediction'].map({1: 'Yes', 0: 'No'})
                    
                    st.write("### Prediction Results:")
                    st.write(results)
                    
                    # Visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Pie chart of predictions
                    churn_counts = results['Churn_Prediction'].value_counts()
                    ax1.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
                    ax1.set_title('Churn Prediction Distribution')
                    
                    # Histogram of probabilities
                    sns.histplot(probabilities, bins=10, ax=ax2)
                    ax2.set_title('Distribution of Churn Probabilities')
                    ax2.set_xlabel('Probability')
                    
                    st.pyplot(fig)
                    
                    # Download link for results
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error making predictions: {e}")
                    st.write("Please ensure your data format matches the model requirements.")

if __name__ == '__main__':
    main()