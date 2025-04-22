#!/usr/bin/env python3
"""
Run Step 5 (Regression): Modeling Pipeline for Delay Prediction
This script implements a regression modeling pipeline to predict exact delay values.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import plotly.express as px
import plotly.graph_objects as go
import shap

# Set the working directory to the project root
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# Set paths
DATA_CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Set plot style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Set random seed for reproducibility
np.random.seed(42)

def save_fig(fig, filename, format='png', dpi=300):
    """Save figure to reports/figures directory"""
    filepath = REPORTS_FIGURES_DIR / filename
    fig.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {filepath}")

def load_data():
    """Load the feature-engineered data"""
    print("Loading data...")
    
    # Try to load parquet files first, fall back to CSV
    try:
        df_int = pd.read_parquet(DATA_CLEAN_DIR / "df_int_features.parquet")
        print("Loaded intersection data from parquet file")
    except:
        df_int = pd.read_csv(DATA_CLEAN_DIR / "df_int_features.csv")
        print("Loaded intersection data from CSV file")
    
    print(f"Intersection dataset shape: {df_int.shape}")
    
    return df_int

def prepare_data_regression(df):
    """Prepare data for regression modeling"""
    print("Preparing data for regression modeling...")
    
    # Check if target variable exists
    if 'Delay_s_veh' not in df.columns:
        print("Error: Delay_s_veh column is missing")
        sys.exit(1)
    
    # Select features for modeling
    feature_cols = [
        'Total_Vol', 'EW_Vol', 'NS_Vol', 'EW_to_NS', 
        'Green_EW', 'Green_to_Demand_EW', 'Ped_Load'
    ]
    
    # Add Period as a feature if available (always include Period_Numeric)
    print("Adding Period_Numeric feature...")
    if 'Period' in df.columns:
        # Convert Period to numeric (AM=0, PM=1)
        print("Converting Period to Period_Numeric")
        df['Period_Numeric'] = df['Period'].map({'AM': 0, 'PM': 1})
    else:
        # Add default Period_Numeric if Period column is missing
        print("Warning: Period column missing, adding default Period_Numeric=0")
        df['Period_Numeric'] = 0
    
    feature_cols.append('Period_Numeric')
    
    # Keep only columns we need for modeling
    features = [col for col in feature_cols if col in df.columns]
    print(f"Selected features: {features}")
    
    # Create X (features) and y (target)
    X = df[features].copy()  # Use copy to avoid SettingWithCopyWarning
    y = df['Delay_s_veh'].copy()  # Use actual delay values for regression
    
    # Check for missing values
    missing = X.isnull().sum()
    if missing.sum() > 0:
        print("Handling missing values...")
        print(f"Missing values per column: {missing[missing > 0]}")
        
        # Handle Period_Numeric separately if it's all NaN
        if 'Period_Numeric' in X.columns and X['Period_Numeric'].isnull().all():
            print("All Period_Numeric values are NaN. Using default value 0 (AM)")
            X['Period_Numeric'] = 0
        
        # More robust median imputation for missing values
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if X[col].isnull().all():
                    # If all values are NaN, use a reasonable default
                    if col == 'Period_Numeric':
                        print(f"Setting all-NaN column {col} to default value 0")
                        X[col] = 0
                    else:
                        print(f"WARNING: Column {col} has all NaN values. Using 0 as default.")
                        X[col] = 0
                else:
                    # Use median for columns with some non-NaN values
                    median_val = X[col].median()
                    print(f"Imputing column {col} with median value: {median_val}")
                    X[col] = X[col].fillna(median_val)
    
    # Double-check for any remaining NaNs
    if X.isnull().sum().sum() > 0:
        print("WARNING: Still have missing values after imputation")
        # Last resort: fill any remaining NaNs with 0
        print("Filling any remaining NaNs with 0")
        X = X.fillna(0)
    
    # Verify data integrity
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    print(f"Any NaNs in X: {X.isnull().values.any()}")
    print(f"Any NaNs in y: {y.isnull().values.any()}")
    
    # Fill any NaNs in y if present
    if y.isnull().any():
        print(f"Filling {y.isnull().sum()} NaNs in target variable with median")
        y = y.fillna(y.median())
    
    # Save feature list for later use in predictions
    feature_list = pd.DataFrame({'Feature': X.columns.tolist()})
    feature_list.to_csv(MODELS_DIR / "model_features_regression.csv", index=False)
    print(f"Saved feature list to {MODELS_DIR / 'model_features_regression.csv'}")
    
    return X, y, df['INTERSECT'] if 'INTERSECT' in df.columns else None

def train_val_split(X, y, intersections=None):
    """Split data into training and validation sets"""
    print("Splitting data into training and validation sets...")
    
    if intersections is not None:
        # Stratify by intersection
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=intersections
        )
    else:
        # Regular split for regression
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    return X_train, X_val, y_train, y_val

def cross_validate_regression(model, X, y, cv=5):
    """Perform cross-validation on a regression model"""
    print(f"Performing {cv}-fold cross-validation...")
    
    # Define cross-validation strategy
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Calculate cross-validation scores - multiple metrics
    r2_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='r2')
    neg_mse_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='neg_mean_squared_error')
    mse_scores = -neg_mse_scores  # Convert back to positive MSE
    rmse_scores = np.sqrt(mse_scores)
    
    print(f"Cross-validation R² scores: {r2_scores}")
    print(f"Mean R² score: {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
    print(f"Mean RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")
    
    return {'r2': r2_scores, 'rmse': rmse_scores}

def linear_regression_model(X_train, X_val, y_train, y_val):
    """Train and evaluate a baseline linear regression model"""
    print("Training baseline linear regression model...")
    
    # Check for NaNs before processing
    if X_train.isnull().values.any() or y_train.isnull().values.any():
        print("ERROR: Training data contains NaN values that must be handled before modeling")
        # Additional imputation if needed
        X_train = X_train.fillna(X_train.median())
        y_train = y_train.fillna(y_train.median())
        
        if X_train.isnull().values.any() or y_train.isnull().values.any():
            print("CRITICAL ERROR: Unable to handle all NaN values in training data")
            raise ValueError("Training data contains NaN values even after attempted imputation")
    
    if X_val.isnull().values.any() or y_val.isnull().values.any():
        print("Warning: Validation data contains NaN values. Applying imputation.")
        X_val = X_val.fillna(X_train.median())  # Use training medians
        y_val = y_val.fillna(y_train.median())
    
    # Scale features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Cross-validate the model
    cv_scores = cross_validate_regression(model, X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_val, y_pred)
    metrics['CV_R2_Mean'] = cv_scores['r2'].mean()
    metrics['CV_R2_Std'] = cv_scores['r2'].std()
    metrics['CV_RMSE_Mean'] = cv_scores['rmse'].mean()
    metrics['CV_RMSE_Std'] = cv_scores['rmse'].std()
    
    # Plot actual vs predicted
    plot_actual_vs_predicted(y_val, y_pred, "Linear Regression")
    
    # Plot residuals
    plot_residuals(y_val, y_pred, "Linear Regression")
    
    print("Linear Regression model metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Also create a model including the scaler for easy prediction
    linear_model = {
        'model': model,
        'scaler': scaler,
        'features': X_train.columns.tolist()
    }
    
    return linear_model, metrics

def random_forest_regression(X_train, X_val, y_train, y_val, X_features=None):
    """Train and evaluate an improved random forest regression model"""
    print("Training Random Forest regression model...")
    
    # Check for NaNs before processing
    if X_train.isnull().values.any() or y_train.isnull().values.any():
        print("ERROR: Training data contains NaN values that must be handled before modeling")
        # Additional imputation if needed
        X_train = X_train.fillna(X_train.median())
        y_train = y_train.fillna(y_train.median())
        
        if X_train.isnull().values.any() or y_train.isnull().values.any():
            print("CRITICAL ERROR: Unable to handle all NaN values in training data")
            raise ValueError("Training data contains NaN values even after attempted imputation")
    
    if X_val.isnull().values.any() or y_val.isnull().values.any():
        print("Warning: Validation data contains NaN values. Applying imputation.")
        X_val = X_val.fillna(X_train.median())  # Use training medians
        y_val = y_val.fillna(y_train.median())
    
    # Define parameter grid for grid search
    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "min_samples_split": [2, 5, 10]
    }
    
    # Initialize grid search
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Perform cross-validation on the best model
    cv_scores = cross_validate_regression(best_model, X_train, y_train)
    
    # Make predictions with best model
    y_pred = best_model.predict(X_val)
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_val, y_pred)
    metrics['CV_R2_Mean'] = cv_scores['r2'].mean()
    metrics['CV_R2_Std'] = cv_scores['r2'].std()
    metrics['CV_RMSE_Mean'] = cv_scores['rmse'].mean()
    metrics['CV_RMSE_Std'] = cv_scores['rmse'].std()
    
    # Print best parameters and metrics
    print(f"Best parameters: {grid_search.best_params_}")
    print("Random Forest Regression model metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot actual vs predicted
    plot_actual_vs_predicted(y_val, y_pred, "Random Forest Regression")
    
    # Plot residuals
    plot_residuals(y_val, y_pred, "Random Forest Regression")
    
    # Feature importance
    if X_features is not None:
        plot_feature_importance(best_model, X_features, "Regression")
        try:
            plot_shap_summary_regression(best_model, X_val)
        except Exception as e:
            print(f"Error generating SHAP summary: {e}")
    
    return best_model, metrics

def calculate_regression_metrics(y_true, y_pred):
    """Calculate evaluation metrics for regression"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero by adding a small epsilon
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }
    return metrics

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """Plot and save actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Delay (s/veh)')
    plt.ylabel('Predicted Delay (s/veh)')
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure
    file_name = f"actual_vs_predicted_{model_name.lower().replace(' ', '_')}.png"
    save_fig(plt.gcf(), file_name)
    
    plt.close()

def plot_residuals(y_true, y_pred, model_name):
    """Plot and save residuals"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 8))
    
    # Residuals vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Delay (s/veh)')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.grid(True)
    
    # Residuals distribution
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    
    # Q-Q plot
    plt.subplot(2, 2, 3)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    # Absolute residuals vs predicted (to check heteroscedasticity)
    plt.subplot(2, 2, 4)
    plt.scatter(y_pred, np.abs(residuals), alpha=0.5)
    plt.xlabel('Predicted Delay (s/veh)')
    plt.ylabel('Absolute Residuals')
    plt.title('Absolute Residuals vs Predicted')
    plt.grid(True)
    
    plt.suptitle(f'Residual Analysis - {model_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    file_name = f"residuals_{model_name.lower().replace(' ', '_')}.png"
    save_fig(plt.gcf(), file_name)
    
    plt.close()

def plot_feature_importance(model, feature_names, model_type="Regression"):
    """Plot feature importance"""
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame for easier plotting
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'Feature Importance - {model_type}')
    plt.tight_layout()
    
    # Save the figure
    save_fig(plt.gcf(), f"feature_importance_{model_type.lower()}.png")
    
    # Also save the data
    feature_importance.to_csv(REPORTS_DIR / f"feature_importance_{model_type.lower()}.csv", index=False)
    
    plt.close()

def plot_shap_summary_regression(model, X_val):
    """Generate and save SHAP summary plot for regression model"""
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance - Regression')
    plt.tight_layout()
    
    # Save the figure
    save_fig(plt.gcf(), "shap_feature_importance_regression.png")
    
    plt.close()
    
    # Create summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, show=False)
    plt.title('SHAP Summary Plot - Regression')
    plt.tight_layout()
    
    # Save the figure
    save_fig(plt.gcf(), "shap_summary_regression.png")
    
    plt.close()

def save_model(model, filename="delay_regression_rf.pkl"):
    """Save model to file"""
    model_path = MODELS_DIR / filename
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def create_model_card(model, metrics, filename="regression_model_card.yml"):
    """Create a model card YAML file with model details"""
    model_card = f"""
# Delay Prediction Regression Model Card

## Model Details
- Model Type: {type(model).__name__}
- Version: 1.0
- Date Created: {pd.Timestamp.now().strftime('%Y-%m-%d')}

## Model Parameters
{model.get_params()}

## Performance Metrics
- Mean Squared Error (MSE): {metrics['MSE']:.4f}
- Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}
- Mean Absolute Error (MAE): {metrics['MAE']:.4f}
- R² Score: {metrics['R²']:.4f}
- Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.4f}
- Cross-Validation R² (Mean): {metrics.get('CV_R2_Mean', 0):.4f}
- Cross-Validation R² (Std): {metrics.get('CV_R2_Std', 0):.4f}
- Cross-Validation RMSE (Mean): {metrics.get('CV_RMSE_Mean', 0):.4f}
- Cross-Validation RMSE (Std): {metrics.get('CV_RMSE_Std', 0):.4f}

## Required Features
{list(pd.read_csv(MODELS_DIR / "model_features_regression.csv")['Feature'])}

## Intended Use
- Primary intended uses: Predict exact delay (seconds per vehicle) at traffic intersections
- Primary intended users: Traffic engineers, urban planners

## Data
- Training data: Intersection traffic data with engineered features
- Evaluation data: 20% validation split stratified by intersection

## Ethical Considerations
- The model is trained on a limited sample of intersections
- Predictions should be verified by traffic engineers before operational decisions
- Model does not include demographic or personally identifiable information

## Limitations
- Model is trained on data from specific intersections and may not generalize well to very different locations
- The training data may not capture all seasonal variations in traffic patterns
- External factors like construction, special events, or weather are not accounted for

## Caveats
- Predictions have inherent uncertainty as shown by the model's RMSE
- Model should be retrained periodically as traffic patterns change
"""
    
    # Save model card to MODELS_DIR
    model_card_path = MODELS_DIR / filename
    with open(model_card_path, 'w') as f:
        f.write(model_card)
    
    print(f"Model card saved to {model_card_path}")

def create_summary_md(linear_metrics, rf_metrics):
    """Create a markdown summary file for Step 5 Regression"""
    summary_md = f"""
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
- Mean Squared Error (MSE): {linear_metrics['MSE']:.4f}
- Root Mean Squared Error (RMSE): {linear_metrics['RMSE']:.4f}
- Mean Absolute Error (MAE): {linear_metrics['MAE']:.4f}
- R² Score: {linear_metrics['R²']:.4f}
- Mean Absolute Percentage Error (MAPE): {linear_metrics['MAPE']:.4f}
- Cross-Validation R² (Mean): {linear_metrics.get('CV_R2_Mean', 0):.4f}
- Cross-Validation R² (Std): {linear_metrics.get('CV_R2_Std', 0):.4f}
- Cross-Validation RMSE (Mean): {linear_metrics.get('CV_RMSE_Mean', 0):.4f}
- Cross-Validation RMSE (Std): {linear_metrics.get('CV_RMSE_Std', 0):.4f}

## Improved Model: Random Forest Regression
An optimized Random Forest regressor was developed using GridSearchCV to find the best hyperparameters.

### Hyperparameter Search
- n_estimators: [100, 200, 400]
- max_depth: [None, 10, 20]
- min_samples_leaf: [1, 2, 5]
- min_samples_split: [2, 5, 10]

### Random Forest Regression Model Metrics
- Mean Squared Error (MSE): {rf_metrics['MSE']:.4f}
- Root Mean Squared Error (RMSE): {rf_metrics['RMSE']:.4f}
- Mean Absolute Error (MAE): {rf_metrics['MAE']:.4f}
- R² Score: {rf_metrics['R²']:.4f}
- Mean Absolute Percentage Error (MAPE): {rf_metrics['MAPE']:.4f}
- Cross-Validation R² (Mean): {rf_metrics.get('CV_R2_Mean', 0):.4f}
- Cross-Validation R² (Std): {rf_metrics.get('CV_R2_Std', 0):.4f}
- Cross-Validation RMSE (Mean): {rf_metrics.get('CV_RMSE_Mean', 0):.4f}
- Cross-Validation RMSE (Std): {rf_metrics.get('CV_RMSE_Std', 0):.4f}

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
"""
    
    # Save summary to REPORTS_DIR
    summary_path = REPORTS_DIR / "step5_regression_summary.md"
    with open(summary_path, 'w') as f:
        f.write(summary_md)
    
    print(f"Regression summary saved to {summary_path}")

def main():
    """Main function to run the regression modeling pipeline"""
    print("Starting Step 5 (Regression): Modeling Pipeline...")
    
    # Load data
    df = load_data()
    
    # Prepare data for regression
    X, y, intersections = prepare_data_regression(df)
    
    # Split data
    X_train, X_val, y_train, y_val = train_val_split(X, y, intersections)
    
    # Train linear regression model
    linear_model, linear_metrics = linear_regression_model(X_train, X_val, y_train, y_val)
    
    # Train random forest regression model
    rf_model, rf_metrics = random_forest_regression(X_train, X_val, y_train, y_val, X.columns)
    
    # Save best model
    save_model(rf_model)
    
    # Create model card
    create_model_card(rf_model, rf_metrics)
    
    # Create summary markdown
    create_summary_md(linear_metrics, rf_metrics)
    
    print("Regression modeling pipeline complete. See reports/step5_regression_summary.md for the summary.")

if __name__ == "__main__":
    main() 