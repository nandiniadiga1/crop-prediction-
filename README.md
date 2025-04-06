🌾 Smart Agriculture: Crop Production Prediction
This project aims to predict the estimated crop production (in tons) for a given region based on multiple real-world agricultural factors. It leverages machine learning techniques to assist farmers, agricultural planners, and policymakers in making informed decisions about crop yields.

📌 Key Features:
Interactive Web App built with Streamlit for user-friendly input and instant predictions.

Inputs include:

📍 State & District

🌱 Season & Crop Type

📅 Crop Year

🌡️ Temperature

💧 Humidity & Soil Moisture

🌾 Area under cultivation

Outputs an accurate production estimate in tons using a trained Random Forest Regressor model.

🔬 Technical Stack:
Python, Pandas, NumPy for data handling

Scikit-learn for:

Label encoding

Feature scaling

Model training & evaluation

Streamlit for building an interactive UI

PIL for image integration to enhance visual appeal

📊 Dataset Info:
Source: Crop Prediction dataset.csv

Includes real agricultural records with attributes like:

State, District, Crop Type, Season, Area

Weather conditions (Temperature, Humidity, Soil Moisture)

Actual Production Output

🤖 Model Details:
Model used: Random Forest Regressor

Trained on cleaned and preprocessed data

Scaled numerical features and encoded categorical variables

Achieved robust performance on test data with good R² scores

🖼️ Interface:
Designed with simplicity and clarity in mind

Includes banner image showcasing smart agriculture

User enters values, clicks Predict, and gets instant output

🌐 Deployment:
Can be deployed on Streamlit Cloud for public access

Lightweight and easily scalable
