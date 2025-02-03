# Bike Rental Sharing Predictions

## Overview
This project leverages machine learning models to predict daily bike rental counts based on weather and seasonal data using the Capital Bike Share system dataset from 2011-2012. The dataset is processed and analyzed using Azure Machine Learning to track experiments and log results.

## Dataset
- **Source:** Capital Bike Share System (2011-2012)
- **Samples:** Daily and hourly rental counts
- **Features:** Weather conditions, seasonality, and other relevant factors
- **Target:** Total daily rental count

Bike-sharing systems offer an eco-friendly transportation alternative and are becoming increasingly popular in urban areas.

## Objective
This project aims to:
- Build and evaluate various machine learning models to predict bike rental demand.
- Identify key influencing factors such as weather and seasonality.
- Determine the most accurate model for predictions.

## Steps in the Notebook
1. **Data Preparation**  
   - Load and preprocess dataset (feature scaling, engineering seasonal data)
   - Split dataset into training and testing sets (70/30 split)
2. **Model Selection**  
   - Train and compare models: Linear Regression, Decision Tree Regression, SGD Regression, and Random Forest Regression
   - Utilize Azure Machine Learning for experiment tracking and logging
3. **Model Evaluation**  
   - Assess models using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared score
   - Compare actual vs. predicted rental counts
4. **Visualization**  
   - Scatter plot of actual vs. predicted values
   - Feature importance analysis
   - Residual analysis for error distribution

## Results Summary
| Model | MSE | RMSE | R-squared |
|--------|---------|---------|-----------|
| Linear Regression | 2,105,638.34 | 1,451.08 | 0.4981 |
| Decision Tree Regression | 2,831,119.21 | 1,682.59 | 0.3251 |
| SGD Regression | 1,999,254.17 | 1,413.95 | 0.5234 |
| **Random Forest Regression** | **1,692,555.79** | **1,300.98** | **0.5965** |

- **Best Model:** Random Forest Regression (highest R², lowest error metrics)
- **Key Insights:**
  - Temperature and seasonality significantly impact bike rental demand.
  - Decision Trees tend to overfit, resulting in poor generalization.
  - Random Forest’s ensemble approach improves predictive accuracy.

## Requirements
- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Azure Machine Learning SDK

## Conclusion
Random Forest Regression is the best-performing model for predicting bike rental demand. Future improvements may include hyperparameter tuning, additional feature engineering, and testing deep learning models.
\