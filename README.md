# Telecom Customer Churn Prediction

This folder contains the final submission for the customer churn prediction assignment.

## Project structure

- data/
  - TelcoCustomerChurn.csv
  - TelcoCustomerChurn - Data Dictionary.csv
- notebook/
  - churn_analysis.ipynb
- model/
  - churn_model.pkl
- app.py
- Data Science Assignment.pdf
- requirements.txt
- README.md
- sample_request.json
- train_model.py

## Local setup

1. Create a Python virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Reference files included in the project:
   - data/TelcoCustomerChurn - Data Dictionary.csv
   - Data Science Assignment.pdf

## Run the project

1. Train the model:
   python train_model.py

2. Start the API:
   uvicorn app:app --reload --host 0.0.0.0 --port 8000

3. Test the API, in new terminal:
   curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample_request.json

## Expected response

The response must contain exactly:
- prediction
- churn_probability

[GitHub repository](https://github.com/Siddharth77/telecom-customer)

## Notes

- The target variable is Churn.
- Train/test split used: 70:30 with random_state = 42.
- The project includes data cleaning, EDA, feature engineering, Decision Tree comparison, evaluation, interpretation, saved model, and API deployment.
