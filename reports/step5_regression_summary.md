
# Step 5: Regression Modeling Pipeline Summary

## Overview
This document summarizes the regression modeling process and results for predicting exact delay values (in seconds per vehicle) at traffic intersections.
The regression models complement the classification models by providing precise delay estimates instead of binary high/low predictions.

## Modeling Approach
The task was to build regression models that predict the exact delay time at intersections based on traffic and signal timing features.

## Data Preparation
- Features selected from the engineered dataset
- Data split into training (80%) and validation (20%) sets
- Stratified sampling by intersection to ensure each intersection is represented in both sets
- Required features are saved to ensure consistency in production

## Baseline Model: Linear Regression
A simple linear regression model was used as the baseline.

### Linear Regression Model Metrics
- Mean Squared Error (MSE): 41.5565
- Root Mean Squared Error (RMSE): 6.4464
- Mean Absolute Error (MAE): 5.4840
- R² Score: 0.8122
- Mean Absolute Percentage Error (MAPE): 15.1384
- Cross-Validation R² (Mean): 0.7918
- Cross-Validation R² (Std): 0.1436
- Cross-Validation RMSE (Mean): 7.9686
- Cross-Validation RMSE (Std): 3.1154

## Improved Model: Random Forest Regression
An optimized Random Forest regressor was developed using GridSearchCV to find the best hyperparameters.

### Hyperparameter Search
- n_estimators: [100, 200, 400]
- max_depth: [None, 10, 20]
- min_samples_leaf: [1, 2, 5]
- min_samples_split: [2, 5, 10]

### Random Forest Regression Model Metrics
- Mean Squared Error (MSE): 3.3010
- Root Mean Squared Error (RMSE): 1.8169
- Mean Absolute Error (MAE): 1.3249
- R² Score: 0.9851
- Mean Absolute Percentage Error (MAPE): 3.3317
- Cross-Validation R² (Mean): 0.9599
- Cross-Validation R² (Std): 0.0318
- Cross-Validation RMSE (Mean): 3.3020
- Cross-Validation RMSE (Std): 0.6530

## Model Evaluation
- Cross-validation was used to assess model stability
- Actual vs. Predicted plots were generated to visualize model accuracy
- Residual analysis was performed to assess model assumptions
- Feature importance analysis was conducted to identify key predictors

## Feature Importance Analysis
- Feature importance was calculated to identify the most influential features
- SHAP (SHapley Additive exPlanations) analysis was performed to interpret model predictions
- The most important features for delay prediction were identified

## Model Persistence
- The best model has been saved for use in predictions
- A model card was created with detailed information about the model
- Required features were saved to ensure consistent prediction inputs

## Comparison with Classification Approach
- Regression models provide precise delay estimates rather than binary classifications
- The Random Forest model outperformed the Linear Regression model in all metrics
- Feature importance rankings are generally consistent between classification and regression approaches
- Regression models provide additional insights into the magnitude of delay

## Next Steps
The regression models complement the classification models and can be used together:
1. Use classification models for binary high/low delay decisions
2. Use regression models for precise delay time estimates
3. Combine both approaches in the interactive dashboard
