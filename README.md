# Churn-Prediction-Model-

Overview
ChurnGuard is a customer churn prediction system for telecom companies, designed to identify customers who are likely to discontinue their services. By analyzing various customer attributes and usage patterns, the system helps businesses proactively implement retention strategies and reduce customer attrition.


Check out the live demo: https://churn-prediction-app-6oz4.onrender.com


Predictive Analytics: Uses machine learning to predict customer churn probability
Single Customer Analysis: Assess individual customers for churn risk
Batch Prediction: Process multiple customers at once via CSV upload
Actionable Recommendations: Provides tailored retention strategies based on churn risk
Interactive Dashboard: Easy-to-use interface for non-technical users
Data Visualization: Visual representation of key churn factors

Project Structure
churn-prediction-model/
├── app.py              # Main Streamlit application with all functionality
├── requirements.txt    # Project dependencies
├── models/             # Directory for trained models
│   └── churnguard_best_model.pkl  # Trained model file
└── README.md           # Project documentation
Installation & Local Setup

Clone the repository:
git clone https://github.com/datamindmedic/churn-prediction-model-.git
cd churn-prediction-model-

Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py

Open your browser and navigate to http://localhost:8501

Usage
Single Customer Prediction

Navigate to the "Single Prediction" tab
Enter customer information in the form
Click "Predict Churn"
View the prediction results and recommendations

Batch Prediction

Navigate to the "Batch Prediction" tab
Upload a CSV file containing customer data
Click "Run Batch Prediction"
Download the results with predictions and recommendations

Model Information
ChurnGuard uses an XGBoost classifier trained on telecom customer data with the following features:

Demographics: Gender, SeniorCitizen, Partner, Dependents
Account Information: Tenure, Contract type, PaperlessBilling, PaymentMethod
Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
Financial: MonthlyCharges, TotalCharges

The model achieves:

Accuracy: 80%+
Precision: 75%+
Recall: 70%+
F1 Score: 72%+
AUC-ROC: 85%+

Deployment
The application is deployed on Streamlit Cloud:

Push your code to GitHub
Go to share.streamlit.io
Connect your GitHub repository
Select the main branch and app.py file
Deploy the application

Future Improvements

Add feature importance visualization
Implement model explainability using SHAP values
Integrate email notification system for high-risk customers
Add more advanced machine learning models
Develop an API endpoint for system integration

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgements

Telecom customer churn dataset
Streamlit for the interactive web framework
XGBoost for the machine learning algorithm
Scikit-learn for model evaluation metrics


Created by [Micheal Omotosho] - [omotoshom11@gmail.com/datamindmedic@gmail.com/@datamindmedic]
