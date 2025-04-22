#!/usr/bin/env python3
"""
Run Step 5: Modeling Pipeline
This script implements the modeling pipeline for traffic delay prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
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

def prepare_data(df):
    """Prepare data for modeling"""
    print("Preparing data for modeling...")
    
    # Check if target variable exists
    if 'High_Delay' not in df.columns:
        print("Creating target variable 'High_Delay'...")
        df['High_Delay'] = (df['Delay_s_veh'] > 60).astype(int)
    
    # Select features for modeling
    feature_cols = [
        'Total_Vol', 'EW_Vol', 'NS_Vol', 'EW_to_NS', 
        'Green_EW', 'Green_to_Demand_EW', 'Ped_Load'
    ]
    
    # Add Period as a feature if available (always include Period_Numeric)
    print("Adding Period_Numeric feature...")
    if 'Period' in df.columns:
        # Convert Period to numeric (AM=0, PM=1)
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
    X = df[features]
    y = df['High_Delay']
    
    # Check for missing values
    missing = X.isnull().sum()
    if missing.sum() > 0:
        print("Handling missing values...")
        # Simple median imputation for missing values
        X = X.fillna(X.median())
    
    # Save feature list for later use in predictions
    feature_list = pd.DataFrame({'Feature': X.columns.tolist()})
    feature_list.to_csv(MODELS_DIR / "model_features.csv", index=False)
    print(f"Saved feature list to {MODELS_DIR / 'model_features.csv'}")
    
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
        # Stratify by target
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y
        )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    return X_train, X_val, y_train, y_val

def cross_validate_model(model, X, y, cv=5):
    """Perform cross-validation on a model"""
    print(f"Performing {cv}-fold cross-validation...")
    
    # Define cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Calculate cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='f1')
    
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean F1 score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    return cv_scores

def baseline_model(X_train, X_val, y_train, y_val):
    """Train and evaluate a baseline decision tree model"""
    print("Training baseline decision tree model...")
    
    # Initialize and train the model
    model0 = DecisionTreeClassifier(random_state=42, max_depth=None)
    model0.fit(X_train, y_train)
    
    # Cross-validate the model
    cv_scores = cross_validate_model(model0, X_train, y_train)
    
    # Make predictions
    y_pred = model0.predict(X_val)
    
    # Calculate metrics
    metrics = calculate_metrics(y_val, y_pred, model0.predict_proba(X_val)[:, 1])
    metrics['CV_F1_Mean'] = cv_scores.mean()
    metrics['CV_F1_Std'] = cv_scores.std()
    
    # Create confusion matrix
    plot_confusion_matrix(y_val, y_pred, "Baseline Decision Tree")
    
    # Generate classification report
    report = classification_report(y_val, y_pred, output_dict=True)
    
    print("Baseline model metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return model0, metrics

def improved_model(X_train, X_val, y_train, y_val, X_features=None):
    """Train and evaluate an improved random forest model using GridSearchCV"""
    print("Training improved random forest model with GridSearchCV...")
    
    # Define parameter grid for grid search - expanded for better performance
    param_grid = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 6, 12, 18],
        "min_samples_leaf": [1, 2, 3, 5],
        "min_samples_split": [2, 5, 10],
        "class_weight": [None, "balanced"]
    }
    
    # Initialize grid search with cross-validation
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Perform cross-validation on the best model
    cv_scores = cross_validate_model(best_model, X_train, y_train)
    
    # Make predictions with best model
    y_pred = best_model.predict(X_val)
    
    # Calculate metrics
    metrics = calculate_metrics(y_val, y_pred, best_model.predict_proba(X_val)[:, 1])
    metrics['CV_F1_Mean'] = cv_scores.mean()
    metrics['CV_F1_Std'] = cv_scores.std()
    
    # Create confusion matrix
    plot_confusion_matrix(y_val, y_pred, "Random Forest (Best Model)")
    
    # Generate ROC curve
    plot_roc_curve(best_model, X_val, y_val, "Random Forest (Best Model)")
    
    # Print best parameters and metrics
    print(f"Best parameters: {grid_search.best_params_}")
    print("Random Forest model metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Feature importance
    if X_features is not None:
        # Create and plot feature selector based on importance threshold
        feature_selector = SelectFromModel(best_model, threshold='mean')
        feature_selector.fit(X_train, y_train)
        
        # Get selected features
        selected_features = X_features[feature_selector.get_support()]
        print(f"Selected features: {selected_features.tolist()}")
        
        # Plot feature importance
        plot_feature_importance(best_model, X_features)
        
        # Check if selected features improve model
        print("Testing model with selected features...")
        X_train_selected = feature_selector.transform(X_train)
        X_val_selected = feature_selector.transform(X_val)
        
        # Train model with selected features
        selected_model = RandomForestClassifier(**grid_search.best_params_, random_state=42)
        selected_model.fit(X_train_selected, y_train)
        
        # Evaluate selected model
        y_pred_selected = selected_model.predict(X_val_selected)
        metrics_selected = calculate_metrics(y_val, y_pred_selected, 
                                            selected_model.predict_proba(X_val_selected)[:, 1])
        
        print("Metrics with selected features:")
        for metric, value in metrics_selected.items():
            print(f"{metric}: {value:.4f}")
        
        # If selected features improve model, use it instead
        if metrics_selected['F1 Score'] > metrics['F1 Score']:
            print("Selected features improved model performance, using selected model.")
            best_model = selected_model
            metrics = metrics_selected
        
        # SHAP analysis
        try:
            plot_shap_summary(best_model, X_val)
            
            # Generate individual SHAP explanations for a few samples
            plot_shap_explanations(best_model, X_val, n_samples=3)
        except Exception as e:
            print(f"Error generating SHAP summary: {e}")
    
    return best_model, metrics

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low Delay', 'High Delay'])
    disp.plot(cmap='Blues', values_format='.2f')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    # Save the figure
    file_name = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    save_fig(plt.gcf(), file_name)
    
    plt.close()

def plot_roc_curve(model, X, y_true, model_name):
    """Plot and save ROC curve"""
    # Get prediction probabilities
    y_prob = model.predict_proba(X)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure
    file_name = f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
    save_fig(plt.gcf(), file_name)
    
    plt.close()

def plot_feature_importance(model, feature_names):
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
    plt.title('Feature Importance')
    plt.tight_layout()
    
    # Save the figure
    save_fig(plt.gcf(), "feature_importance.png")
    
    # Also save the data
    feature_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    
    plt.close()

def plot_shap_summary(model, X_val):
    """Generate and save SHAP summary plot"""
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    
    # For classification, shap_values is a list of arrays (one per class)
    # We take the second array which corresponds to the positive class (High Delay)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    
    # Save the figure
    save_fig(plt.gcf(), "shap_feature_importance.png")
    
    plt.close()
    
    # Create summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    
    # Save the figure
    save_fig(plt.gcf(), "shap_summary.png")
    
    plt.close()

def plot_shap_explanations(model, X_val, n_samples=3):
    """Generate individual SHAP explanations for a few samples"""
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    
    # Select a few samples
    indices = np.random.choice(range(len(X_val)), min(n_samples, len(X_val)), replace=False)
    
    for i, idx in enumerate(indices):
        sample = X_val.iloc[idx:idx+1]
        
        # Calculate SHAP values for this sample
        shap_values = explainer.shap_values(sample)
        
        # For classification, shap_values is a list of arrays (one per class)
        # We take the second array which corresponds to the positive class (High Delay)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create plot
        plt.figure(figsize=(12, 4))
        shap.force_plot(
            explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
            shap_values,
            sample,
            matplotlib=True,
            show=False
        )
        plt.title(f'SHAP Explanation - Sample {i+1}')
        plt.tight_layout()
        
        # Save the figure
        save_fig(plt.gcf(), f"shap_explanation_sample_{i+1}.png")
        
        plt.close()

def save_model(model, filename="high_delay_rf.pkl"):
    """Save model to file"""
    model_path = MODELS_DIR / filename
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def create_model_card(model, metrics, filename="model_card.yml"):
    """Create a model card YAML file with model details"""
    model_card = f"""
# High Delay Prediction Model Card

## Model Details
- Model Type: {type(model).__name__}
- Version: 1.0
- Date Created: {pd.Timestamp.now().strftime('%Y-%m-%d')}

## Model Parameters
{model.get_params()}

## Performance Metrics
- Accuracy: {metrics['Accuracy']:.4f}
- Precision: {metrics['Precision']:.4f}
- Recall: {metrics['Recall']:.4f}
- F1 Score: {metrics['F1 Score']:.4f}
- ROC-AUC: {metrics['ROC-AUC']:.4f}
- Cross-Validation F1 (Mean): {metrics.get('CV_F1_Mean', 0):.4f}
- Cross-Validation F1 (Std): {metrics.get('CV_F1_Std', 0):.4f}

## Required Features
{list(pd.read_csv(MODELS_DIR / "model_features.csv")['Feature'])}

## Intended Use
- Primary intended uses: Predict high delay (>60 seconds) at traffic intersections
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
- Predictions are probabilistic and should be used as guidance rather than absolute truth
- Model should be retrained periodically as traffic patterns change
"""
    
    # Save model card to MODELS_DIR
    model_card_path = MODELS_DIR / filename
    with open(model_card_path, 'w') as f:
        f.write(model_card)
    
    print(f"Model card saved to {model_card_path}")

def create_summary_md(baseline_metrics, improved_metrics):
    """Create a markdown summary file for Step 5"""
    summary_md = f"""
# Step 5: Modeling Pipeline Summary

## Overview
This document summarizes the modeling process and results for predicting high delay at traffic intersections.
The modeling follows the exploratory data analysis completed in Step 4.

## Modeling Approach
The task was to build a classification model that predicts whether an intersection will experience high delay 
(>60 seconds per vehicle) based on traffic and signal timing features.

## Data Preparation
- Features selected from the engineered dataset
- Data split into training (80%) and validation (20%) sets
- Stratified sampling by intersection to ensure each intersection is represented in both sets
- Required features are saved to ensure consistency in production

## Baseline Model: Decision Tree
A simple decision tree classifier was used as the baseline model.

### Baseline Model Metrics
- Accuracy: {baseline_metrics['Accuracy']:.4f}
- Precision: {baseline_metrics['Precision']:.4f}
- Recall: {baseline_metrics['Recall']:.4f}
- F1 Score: {baseline_metrics['F1 Score']:.4f}
- ROC-AUC: {baseline_metrics['ROC-AUC']:.4f}
- Cross-Validation F1 (Mean): {baseline_metrics.get('CV_F1_Mean', 0):.4f}
- Cross-Validation F1 (Std): {baseline_metrics.get('CV_F1_Std', 0):.4f}

## Improved Model: Random Forest
An optimized Random Forest classifier was developed using GridSearchCV to find the best hyperparameters.

### Hyperparameter Search
- n_estimators: [200, 400, 800]
- max_depth: [None, 6, 12, 18]
- min_samples_leaf: [1, 2, 3, 5]
- min_samples_split: [2, 5, 10]
- class_weight: [None, "balanced"]

### Improved Model Metrics
- Accuracy: {improved_metrics['Accuracy']:.4f}
- Precision: {improved_metrics['Precision']:.4f}
- Recall: {improved_metrics['Recall']:.4f}
- F1 Score: {improved_metrics['F1 Score']:.4f}
- ROC-AUC: {improved_metrics['ROC-AUC']:.4f}
- Cross-Validation F1 (Mean): {improved_metrics.get('CV_F1_Mean', 0):.4f}
- Cross-Validation F1 (Std): {improved_metrics.get('CV_F1_Std', 0):.4f}

## Model Evaluation
- Cross-validation was used to assess model stability
- Confusion matrices were generated to visualize classification performance
- ROC curves were plotted to evaluate classifier thresholds
- Feature selection was explored to identify the most important variables

## Feature Importance Analysis
- Feature importance was calculated to identify the most influential features
- SHAP (SHapley Additive exPlanations) analysis was performed to interpret model predictions
- Individual explanations were generated to understand specific prediction cases

## Model Persistence
- The best model has been saved as `models/high_delay_rf.pkl` for use in predictions
- A model card was created with detailed information about the model
- Required features were saved to ensure consistent prediction inputs

## Next Steps
The next step (Step 6) will focus on scenario generation and transfer learning:
1. Using link data to create synthetic scenarios
2. Predicting delay probabilities for these scenarios
3. Identifying conditions that lead to reduced delay
"""
    
    # Save summary to REPORTS_DIR
    summary_path = REPORTS_DIR / "step5_summary.md"
    with open(summary_path, 'w') as f:
        f.write(summary_md)
    
    print(f"Summary saved to {summary_path}")

def main():
    """Main function to run the modeling pipeline"""
    print("Starting Step 5: Modeling Pipeline...")
    
    # Load data
    df = load_data()
    
    # Prepare data
    X, y, intersections = prepare_data(df)
    
    # Split data
    X_train, X_val, y_train, y_val = train_val_split(X, y, intersections)
    
    # Train baseline model
    baseline_model_obj, baseline_metrics = baseline_model(X_train, X_val, y_train, y_val)
    
    # Train improved model
    improved_model_obj, improved_metrics = improved_model(X_train, X_val, y_train, y_val, X.columns)
    
    # Save best model
    save_model(improved_model_obj)
    
    # Create model card
    create_model_card(improved_model_obj, improved_metrics)
    
    # Create summary markdown
    create_summary_md(baseline_metrics, improved_metrics)
    
    print("Modeling pipeline complete. See reports/step5_summary.md for the summary.")

if __name__ == "__main__":
    main() 