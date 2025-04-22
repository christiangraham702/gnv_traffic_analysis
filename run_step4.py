#!/usr/bin/env python3
"""
Run Step 4: Exploratory Data Analysis
This script performs exploratory data analysis on the traffic data.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Set the working directory to the project root
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# Set paths
DATA_CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create reports directory if it doesn't exist
REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
        df_link = pd.read_parquet(DATA_CLEAN_DIR / "df_link_features.parquet")
        print("Loaded data from parquet files")
    except:
        df_int = pd.read_csv(DATA_CLEAN_DIR / "df_int_features.csv")
        df_link = pd.read_csv(DATA_CLEAN_DIR / "df_link_features.csv")
        print("Loaded data from CSV files")
    
    print(f"Intersection dataset shape: {df_int.shape}")
    print(f"Link dataset shape: {df_link.shape}")
    
    return df_int, df_link

def univariate_analysis(df_int, df_link):
    """Perform univariate analysis on the datasets"""
    print("Performing univariate analysis...")
    
    # Intersection dataset - examine key features
    numeric_cols = ['Total_Vol', 'EW_Vol', 'NS_Vol', 'EW_to_NS', 'Green_EW', 
                   'Green_to_Demand_EW', 'Ped_Load', 'Delay_s_veh']
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if col in df_int.columns:
            sns.histplot(df_int[col].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_ylabel('Count')
        else:
            axes[i].set_title(f'{col} not found')
    
    plt.tight_layout()
    save_fig(fig, "intersection_feature_distributions.png")
    
    # Link dataset - examine key features
    numeric_cols = ['Peak_Ratio', 'Direction_Balance', 'AM_PM_Diff', 
                   'AM_Peak_Ratio', 'PM_Peak_Ratio', 'HvyPct_1314']
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if col in df_link.columns:
            sns.histplot(df_link[col].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_ylabel('Count')
        else:
            axes[i].set_title(f'{col} not found')
    
    plt.tight_layout()
    save_fig(fig, "link_feature_distributions.png")

def bivariate_analysis(df_int, df_link):
    """Perform bivariate analysis on the datasets"""
    print("Performing bivariate analysis...")
    
    # Create scatter plots of Delay vs. other features
    feature_cols = ['Total_Vol', 'EW_Vol', 'NS_Vol', 'EW_to_NS', 'Green_EW', 'Ped_Load']
    target_col = 'Delay_s_veh'
    
    if target_col in df_int.columns:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(feature_cols):
            if feature in df_int.columns:
                sns.scatterplot(x=feature, y=target_col, data=df_int, 
                                hue='INTERSECT' if 'INTERSECT' in df_int.columns else None, 
                                alpha=0.7, ax=axes[i])
                axes[i].set_title(f'{feature} vs {target_col}')
            else:
                axes[i].set_title(f'{feature} not found')
        
        plt.tight_layout()
        save_fig(fig, "features_vs_delay.png")
    
    # Boxplots of features by High_Delay category
    if 'High_Delay' in df_int.columns:
        feature_cols = ['Total_Vol', 'EW_Vol', 'NS_Vol', 'EW_to_NS', 'Green_EW', 'Green_to_Demand_EW']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(feature_cols):
            if feature in df_int.columns:
                sns.boxplot(x='High_Delay', y=feature, data=df_int, ax=axes[i])
                axes[i].set_title(f'{feature} by Delay Category')
                axes[i].set_xlabel('High Delay (1=Yes, 0=No)')
            else:
                axes[i].set_title(f'{feature} not found')
        
        plt.tight_layout()
        save_fig(fig, "features_by_delay_category.png")
    
    # Scatter plot of AM vs PM Peak for link data
    if all(col in df_link.columns for col in ['AM_Peak_Ratio', 'PM_Peak_Ratio']):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.scatterplot(x='AM_Peak_Ratio', y='PM_Peak_Ratio', data=df_link, 
                       alpha=0.7, ax=ax)
        
        # Add a 45-degree reference line
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        limit = max(xmax, ymax)
        ax.plot([0, limit], [0, limit], 'k--', alpha=0.5, label='1:1 Line')
        
        ax.set_title('AM vs PM Peak Ratio Comparison')
        ax.set_xlabel('AM Peak Ratio (Proportion of ADT)')
        ax.set_ylabel('PM Peak Ratio (Proportion of ADT)')
        ax.legend()
        
        plt.tight_layout()
        save_fig(fig, "am_vs_pm_peak.png")

def correlation_analysis(df_int, df_link):
    """Perform correlation analysis on the datasets"""
    print("Performing correlation analysis...")
    
    # Intersection dataset correlation matrix
    int_corr_cols = ['Total_Vol', 'EW_Vol', 'NS_Vol', 'EW_to_NS', 'Green_EW', 
                   'Green_to_Demand_EW', 'Ped_Load', 'Delay_s_veh']
    
    # Filter to include only columns that exist in the dataframe
    int_corr_cols = [col for col in int_corr_cols if col in df_int.columns]
    
    if int_corr_cols:
        # Compute correlation matrix
        corr_matrix = df_int[int_corr_cols].corr()
        
        # Plot the correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                   annot=True, fmt='.2f', square=True, linewidths=.5,
                   cbar_kws={"shrink": .5})
        
        plt.title('Correlation Matrix of Intersection Features')
        plt.tight_layout()
        save_fig(plt.gcf(), "intersection_correlation_matrix.png")
    
    # Link dataset correlation matrix
    link_corr_cols = ['Peak_Ratio', 'Direction_Balance', 'AM_PM_Diff', 
                     'AM_Peak_Ratio', 'PM_Peak_Ratio', 'HvyPct_1314', 'ADT_1314']
    
    # Filter to include only columns that exist in the dataframe
    link_corr_cols = [col for col in link_corr_cols if col in df_link.columns]
    
    if link_corr_cols:
        # Compute correlation matrix
        corr_matrix = df_link[link_corr_cols].corr()
        
        # Plot the correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                   annot=True, fmt='.2f', square=True, linewidths=.5,
                   cbar_kws={"shrink": .5})
        
        plt.title('Correlation Matrix of Link Features')
        plt.tight_layout()
        save_fig(plt.gcf(), "link_correlation_matrix.png")

def period_analysis(df_int):
    """Analyze metrics by time period (AM/PM)"""
    print("Performing period analysis...")
    
    # Analyze metrics by Period (AM/PM)
    if 'Period' in df_int.columns:
        metrics = ['Total_Vol', 'Delay_s_veh']
        metrics = [m for m in metrics if m in df_int.columns]
        
        if metrics:
            fig, axes = plt.subplots(1, len(metrics), figsize=(15, 6))
            if len(metrics) == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                sns.boxplot(x='Period', y=metric, data=df_int, ax=axes[i])
                axes[i].set_title(f'{metric} by Period')
            
            plt.tight_layout()
            save_fig(fig, "metrics_by_period.png")

def main():
    """Main function to run the EDA process"""
    print("Starting Step 4: Exploratory Data Analysis...")
    
    # Load data
    df_int, df_link = load_data()
    
    # Perform analyses
    univariate_analysis(df_int, df_link)
    bivariate_analysis(df_int, df_link)
    correlation_analysis(df_int, df_link)
    period_analysis(df_int)
    
    # Write the summary document (already created separately)
    print("EDA complete. See reports/step4_summary.md for the summary.")

if __name__ == "__main__":
    main() 