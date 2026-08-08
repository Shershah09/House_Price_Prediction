# 🏠 House Price Prediction

A Machine Learning web application that predicts Boston house prices using a Gradient Boosting Regressor and Streamlit.

## 📌 Project Overview

This project uses the Boston Housing Dataset to predict the estimated price of a house based on different housing and socioeconomic features.

The trained Machine Learning model is integrated into a Streamlit web application where users can enter house details and get a predicted house price.

## 🤖 Machine Learning Model

- Gradient Boosting Regressor
- Hyperparameter tuning using GridSearchCV
- R² Score: **0.881**
- RMSE: **2.998**

## 📊 Features

The model uses the following features:

- CRIM – Crime Rate
- ZN – Residential Land
- INDUS – Industrial Area
- CHAS – Charles River
- NOX – Nitric Oxide
- RM – Average Rooms
- AGE – House Age
- DIS – Distance
- RAD – Highway Access
- TAX – Property Tax
- PTRATIO – Student-Teacher Ratio
- B – Black Population Index
- LSTAT – Lower Status Population

## 🔄 Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Data Preprocessing
5. Train-Test Split
6. Pipeline & ColumnTransformer
7. Model Training
8. Hyperparameter Tuning
9. Model Evaluation
10. Streamlit Deployment

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## 🚀 How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
