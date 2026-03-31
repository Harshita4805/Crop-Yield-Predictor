# Crop Yield Prediction using Machine Learning

## Overview
A machine learning project that predicts crop yield (kg/hectare) based on
state, crop type, season, area, and year.

## Models Used
- Linear Regression (baseline)
- Random Forest (best performer)
- XGBoost

## Results
| Model              | R² Score | RMSE  |
|--------------------|----------|-------|
| Linear Regression  | ~0.65    | high  |
| Random Forest      | ~0.88    | low   |
| XGBoost            | ~0.86    | low   |

## How to Run
pip install -r requirements.txt
python crop_yield.py       # train the model
streamlit run app.py       # launch web app

## Dataset
Crop Production in India — Kaggle
