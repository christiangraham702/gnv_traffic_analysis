#!/usr/bin/env python
"""
Run Step 3: Feature Engineering
This script creates derived features from the cleaned data for both intersection and link datasets.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the random seed for reproducibility
np.random.seed(42)

# Set up paths - use absolute paths for reliability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CLEAN_DIR = os.path.join(SCRIPT_DIR, "data", "clean")
REPORTS_FIGURES_DIR = os.path.join(SCRIPT_DIR, "reports", "figures")

# Make sure directories exist
os.makedirs(REPORTS_FIGURES_DIR, exist_ok=True)

def load_clean_data():
    """Load the cleaned data files."""
    # Try to load CSV files first (most reliable)
    df_int = pd.read_csv(os.path.join(DATA_CLEAN_DIR, "df_int.csv"))
    df_link = pd.read_csv(os.path.join(DATA_CLEAN_DIR, "df_link.csv"))
    
    # Convert timestamp column to datetime if exists
    if 'Timestamp' in df_int.columns:
        df_int['Timestamp'] = pd.to_datetime(df_int['Timestamp'], errors='coerce')
    
    return df_int, df_link

def engineer_intersection_features(df_int):
    """
    Create engineered features for intersection data.
    
    Features to create:
    - Total_Vol: sum of all turning-movement counts
    - EW_Vol: (EB_L + EB_Th + EB_R) + (WB_L + WB_Th + WB_R)
    - NS_Vol: (NB_L + NB_Th + NB_R) + (SB_L + SB_Th + SB_R)
    - EW_to_NS: EW_Vol / NS_Vol
    - Green_EW: Max_green_EBL + Max_green_EBT (or mean)
    - Green_to_Demand_EW: Green_EW / EW_Vol
    - Ped_Load: Ped_clearance_EW + Ped_clearance_NS
    - High_Delay (target): Delay_s_veh > 60
    """
    print("\nEngineering intersection features...")
    
    # Check if we have the necessary columns for feature engineering
    turn_cols = ['EB_L', 'EB_Th', 'EB_R', 'WB_L', 'WB_Th', 'WB_R', 
                 'NB_L', 'NB_Th', 'NB_R', 'SB_L', 'SB_Th', 'SB_R']
    
    if not all(col in df_int.columns for col in turn_cols):
        print("Warning: Some turning movement columns are missing. Feature engineering may be incomplete.")
    
    # Feature: Total_Vol - sum of all turning movements
    df_int['Total_Vol'] = df_int[turn_cols].sum(axis=1)
    
    # Feature: EW_Vol - eastbound + westbound volumes
    df_int['EW_Vol'] = df_int[['EB_L', 'EB_Th', 'EB_R', 'WB_L', 'WB_Th', 'WB_R']].sum(axis=1)
    
    # Feature: NS_Vol - northbound + southbound volumes
    df_int['NS_Vol'] = df_int[['NB_L', 'NB_Th', 'NB_R', 'SB_L', 'SB_Th', 'SB_R']].sum(axis=1)
    
    # Feature: EW_to_NS - ratio of east-west to north-south volumes
    # Avoid division by zero
    df_int['EW_to_NS'] = df_int['EW_Vol'] / df_int['NS_Vol'].replace(0, np.nan)
    
    # Feature: Green_EW - eastbound green time
    green_ew_cols = ['Max_green_EBL', 'Max_green_EBT']
    if all(col in df_int.columns for col in green_ew_cols):
        df_int['Green_EW'] = df_int[green_ew_cols].mean(axis=1)
    else:
        print("Warning: Eastbound green time columns missing. Skipping Green_EW feature.")
    
    # Feature: Green_to_Demand_EW - ratio of green time to demand
    if 'Green_EW' in df_int.columns:
        df_int['Green_to_Demand_EW'] = df_int['Green_EW'] / df_int['EW_Vol'].replace(0, np.nan)
    
    # Feature: Ped_Load - pedestrian clearance time
    ped_cols = ['Ped_clearence_EW', 'Ped_clearence_NS']
    if all(col in df_int.columns for col in ped_cols):
        df_int['Ped_Load'] = df_int[ped_cols].sum(axis=1)
    else:
        print("Warning: Pedestrian clearance columns missing. Skipping Ped_Load feature.")
    
    # Create target column: High_Delay (binary)
    if 'Delay_s_veh' in df_int.columns:
        df_int['High_Delay'] = (df_int['Delay_s_veh'] > 60).astype(int)
    else:
        print("Warning: Delay_s_veh column missing. Cannot create High_Delay target variable.")
    
    # Print summary of new features
    new_features = ['Total_Vol', 'EW_Vol', 'NS_Vol', 'EW_to_NS', 
                    'Green_EW', 'Green_to_Demand_EW', 'Ped_Load', 'High_Delay']
    print("Created intersection features:")
    for feature in new_features:
        if feature in df_int.columns:
            print(f"  - {feature}: {df_int[feature].notna().sum()} non-null values")
    
    return df_int

def engineer_link_features(df_link):
    """
    Create engineered features for link dataset.
    
    Features to create:
    - Restrict to reference year (2013-2014)
    - Peak_Ratio: PkPM_1314 / PkAM_1314
    - Direction balance features
    """
    print("\nEngineering link features...")
    
    # Check if we have the necessary columns
    ref_year_cols = ['PkPM_1314', 'PkAM_1314', 'ADT_1314']
    if not all(col in df_link.columns for col in ref_year_cols):
        print("Warning: Reference year columns missing. Feature engineering may be incomplete.")
    
    # Feature: Peak_Ratio - ratio of PM to AM peak traffic
    if all(x in df_link.columns for x in ['PkPM_1314', 'PkAM_1314']):
        df_link['Peak_Ratio'] = df_link['PkPM_1314'] / df_link['PkAM_1314'].replace(0, np.nan)
    
    # Feature: Direction_Balance - difference between direction 1 and 2 proportions
    if all(x in df_link.columns for x in ['DlyD1_1314', 'DlyD2_1314']):
        df_link['Direction_Balance'] = abs(df_link['DlyD1_1314'] - df_link['DlyD2_1314'])
    
    # Feature: AM_PM_Diff - difference between AM and PM peaks
    if all(x in df_link.columns for x in ['PkPM_1314', 'PkAM_1314']):
        df_link['AM_PM_Diff'] = df_link['PkPM_1314'] - df_link['PkAM_1314']
    
    # Feature: Normalize AM/PM peaks by ADT
    if 'ADT_1314' in df_link.columns:
        if 'PkAM_1314' in df_link.columns:
            df_link['AM_Peak_Ratio'] = df_link['PkAM_1314'] / df_link['ADT_1314'].replace(0, np.nan)
        if 'PkPM_1314' in df_link.columns:
            df_link['PM_Peak_Ratio'] = df_link['PkPM_1314'] / df_link['ADT_1314'].replace(0, np.nan)
    
    # Print summary of new features
    new_features = ['Peak_Ratio', 'Direction_Balance', 'AM_PM_Diff', 'AM_Peak_Ratio', 'PM_Peak_Ratio']
    print("Created link features:")
    for feature in new_features:
        if feature in df_link.columns:
            print(f"  - {feature}: {df_link[feature].notna().sum()} non-null values")
    
    return df_link

def visualize_features(df_int, df_link):
    """Create visualizations of key engineered features."""
    print("\nCreating feature visualizations...")
    
    # Create figures directory if it doesn't exist
    os.makedirs(REPORTS_FIGURES_DIR, exist_ok=True)
    
    # Figure 1: Intersection volume distributions
    if all(x in df_int.columns for x in ['Total_Vol', 'EW_Vol', 'NS_Vol']):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df_int, x='Total_Vol', kde=True)
        plt.title('Distribution of Total Volume at Intersections')
        plt.xlabel('Total Volume')
        plt.savefig(os.path.join(REPORTS_FIGURES_DIR, 'total_volume_dist.png'))
        plt.close()
    
    # Figure 2: High Delay vs Total Volume
    if all(x in df_int.columns for x in ['High_Delay', 'Total_Vol']):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_int, x='High_Delay', y='Total_Vol')
        plt.title('Total Volume by Delay Category')
        plt.xlabel('High Delay (1=Yes, 0=No)')
        plt.ylabel('Total Volume')
        plt.savefig(os.path.join(REPORTS_FIGURES_DIR, 'delay_vs_volume.png'))
        plt.close()
    
    # Figure 3: Correlation matrix for intersection features
    int_features = ['Total_Vol', 'EW_Vol', 'NS_Vol', 'EW_to_NS', 
                    'Green_EW', 'Green_to_Demand_EW', 'Delay_s_veh', 'High_Delay']
    int_features = [f for f in int_features if f in df_int.columns]
    
    if len(int_features) > 1:
        plt.figure(figsize=(12, 10))
        corr = df_int[int_features].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of Intersection Features')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_FIGURES_DIR, 'intersection_correlation.png'))
        plt.close()
    
    # Figure 4: Link Peak Ratio distribution
    if 'Peak_Ratio' in df_link.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df_link, x='Peak_Ratio', kde=True)
        plt.title('Distribution of Peak Ratio (PM/AM) on Links')
        plt.xlabel('Peak Ratio (PM/AM)')
        plt.savefig(os.path.join(REPORTS_FIGURES_DIR, 'peak_ratio_dist.png'))
        plt.close()
    
    print(f"Visualizations saved to {REPORTS_FIGURES_DIR}")

def save_engineered_data(df_int, df_link):
    """Save the datasets with engineered features."""
    print("\nSaving engineered data...")
    
    # Save as CSV (always works)
    df_int.to_csv(os.path.join(DATA_CLEAN_DIR, "df_int_features.csv"), index=False)
    df_link.to_csv(os.path.join(DATA_CLEAN_DIR, "df_link_features.csv"), index=False)
    
    # Try to save to parquet if pyarrow is available
    try:
        import pyarrow
        df_int.to_parquet(os.path.join(DATA_CLEAN_DIR, "df_int_features.parquet"))
        df_link.to_parquet(os.path.join(DATA_CLEAN_DIR, "df_link_features.parquet"))
        print("Data saved in both CSV and Parquet formats")
    except ImportError:
        print("Pyarrow not available. Data saved in CSV format only.")
    
    print(f"Engineered data saved to {DATA_CLEAN_DIR}")
    print(f"  - Intersection data with features: {df_int.shape} rows x columns")
    print(f"  - Link data with features: {df_link.shape} rows x columns")

def main():
    print("Running Step 3: Feature Engineering")
    
    # Load data
    print("\nLoading cleaned data...")
    df_int, df_link = load_clean_data()
    print(f"Loaded intersection data: {df_int.shape}")
    print(f"Loaded link data: {df_link.shape}")
    
    # Create engineered features
    df_int = engineer_intersection_features(df_int)
    df_link = engineer_link_features(df_link)
    
    # Create visualizations
    visualize_features(df_int, df_link)
    
    # Save engineered data
    save_engineered_data(df_int, df_link)
    
    print("\nStep 3 completed successfully!")

if __name__ == "__main__":
    main() 