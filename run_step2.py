#!/usr/bin/env python
"""
Run Step 2: Data Cleaning & Harmonisation
This script performs data cleaning and harmonization on the traffic datasets.
"""

import os
import pandas as pd
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Set up paths
data_raw_path = "data/raw/"
data_clean_path = "data/clean/"
reports_tables_path = "reports/tables/"

# Make sure directories exist
os.makedirs(data_clean_path, exist_ok=True)
os.makedirs(reports_tables_path, exist_ok=True)

def clean_percentage(value):
    """Convert percentage string to float."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str) and '%' in value:
        return float(value.strip('%')) / 100
    return value

def main():
    print("Running Step 2: Data Cleaning & Harmonisation")
    
    # 2.1 Load Files
    print("\nLoading raw data files...")
    df_int = pd.read_csv(os.path.join(data_raw_path, "traffic_data.csv"))
    print(f"Intersection data shape: {df_int.shape}")
    
    df_link = pd.read_csv(os.path.join(data_raw_path, "lots_traffic_data.csv"))
    print(f"Link data shape: {df_link.shape}")
    
    # 2.2 Type Coercion
    print("\nPerforming type coercion...")
    
    # Clean heavy vehicle columns
    heavy_cols = [col for col in df_link.columns if 'Heavy' in col]
    print(f"Found {len(heavy_cols)} heavy vehicle columns")
    
    for col in heavy_cols:
        df_link[col] = df_link[col].apply(clean_percentage).astype(float)
    
    # Convert green times and lane counts to numeric
    green_cols = [col for col in df_int.columns if 'green' in col.lower()]
    lane_cols = [col for col in df_int.columns if 'lanes' in col.lower()]
    
    print(f"Found {len(green_cols)} green time columns and {len(lane_cols)} lane count columns")
    
    for col in green_cols + lane_cols:
        df_int[col] = pd.to_numeric(df_int[col], errors='coerce')
    
    # 2.3 Datetime Conversion
    print("\nCreating timestamp column...")
    
    # Add a reference year if 'Year' isn't present
    if 'Year' not in df_int.columns:
        df_int['Year'] = 2014
    
    # Create a timestamp column if Period and Time are available
    if 'Period' in df_int.columns and 'Time' in df_int.columns:
        # Map periods to months (assuming AM/PM periods correspond to different times of year)
        period_map = {'AM': '01', 'PM': '07'}  # January for AM, July for PM (arbitrary)
        
        # Create a datetime string
        df_int['datetime_str'] = df_int['Year'].astype(str) + '-' + \
                                 df_int['Period'].map(period_map) + '-01 ' + \
                                 df_int['Time']
        
        # Convert to datetime
        df_int['Timestamp'] = pd.to_datetime(df_int['datetime_str'], errors='coerce')
        
        # Remove the intermediate string column
        df_int.drop('datetime_str', axis=1, inplace=True)
        
        print("Timestamp column created successfully")
    
    # 2.4 Missing Values
    print("\nHandling missing values...")
    
    # Check missing values
    int_missing = df_int.isnull().sum()
    int_missing_pct = int_missing / len(df_int) * 100
    
    link_missing = df_link.isnull().sum()
    link_missing_pct = link_missing / len(df_link) * 100
    
    # Define critical columns
    int_critical_cols = ['INTERSECT', 'Period', 'Time', 'Delay_s_veh']
    link_critical_cols = ['Station', 'STREET', 'the_geom']
    
    # Initialize dataframes to track dropped rows
    dropped_int_rows = pd.DataFrame()
    dropped_link_rows = pd.DataFrame()
    
    # Process intersection data
    for col in df_int.columns:
        missing_pct = int_missing_pct[col]
        
        if missing_pct == 0:  # No missing values
            continue
        
        if missing_pct < 5 and col not in int_critical_cols:
            # Impute with median for numeric columns
            if pd.api.types.is_numeric_dtype(df_int[col]):
                median_val = df_int[col].median()
                df_int[col].fillna(median_val, inplace=True)
                print(f"  Imputed {col} with median: {median_val}")
            else:
                # For non-numeric, use mode (most frequent value)
                mode_val = df_int[col].mode()[0]
                df_int[col].fillna(mode_val, inplace=True)
                print(f"  Imputed {col} with mode: {mode_val}")
        else:
            # Mark rows with missing values in critical columns
            rows_to_drop = df_int[df_int[col].isnull()].copy()
            if not rows_to_drop.empty:
                rows_to_drop['drop_reason'] = f"Missing {col}"
                dropped_int_rows = pd.concat([dropped_int_rows, rows_to_drop])
                print(f"  Marked {len(rows_to_drop)} rows for dropping due to missing {col}")
    
    # Process link data
    for col in df_link.columns:
        missing_pct = link_missing_pct[col]
        
        if missing_pct == 0:  # No missing values
            continue
        
        if missing_pct < 5 and col not in link_critical_cols:
            # Impute with median for numeric columns
            if pd.api.types.is_numeric_dtype(df_link[col]):
                median_val = df_link[col].median()
                df_link[col].fillna(median_val, inplace=True)
                print(f"  Imputed {col} with median: {median_val}")
            else:
                # For non-numeric, use mode (most frequent value)
                mode_val = df_link[col].mode()[0]
                df_link[col].fillna(mode_val, inplace=True)
                print(f"  Imputed {col} with mode: {mode_val}")
        else:
            # Mark rows with missing values in critical columns
            rows_to_drop = df_link[df_link[col].isnull()].copy()
            if not rows_to_drop.empty:
                rows_to_drop['drop_reason'] = f"Missing {col}"
                dropped_link_rows = pd.concat([dropped_link_rows, rows_to_drop])
                print(f"  Marked {len(rows_to_drop)} rows for dropping due to missing {col}")
    
    # Drop the rows marked for deletion
    if not dropped_int_rows.empty:
        dropped_int_rows = dropped_int_rows.drop_duplicates()
        original_count = len(df_int)
        df_int = df_int.dropna(subset=int_critical_cols)
        print(f"  Dropped {original_count - len(df_int)} rows from intersection data")
    
    if not dropped_link_rows.empty:
        dropped_link_rows = dropped_link_rows.drop_duplicates()
        original_count = len(df_link)
        df_link = df_link.dropna(subset=link_critical_cols)
        print(f"  Dropped {original_count - len(df_link)} rows from link data")
    
    # Save dropped rows to reports/tables/dropped_rows.csv
    if not dropped_int_rows.empty or not dropped_link_rows.empty:
        # Save to CSV
        if not dropped_int_rows.empty:
            dropped_int_rows.to_csv(os.path.join(reports_tables_path, "dropped_int_rows.csv"), index=False)
        
        if not dropped_link_rows.empty:
            dropped_link_rows.to_csv(os.path.join(reports_tables_path, "dropped_link_rows.csv"), index=False)
        
        # Create a summary markdown file
        with open(os.path.join(reports_tables_path, "dropped_rows_summary.md"), "w") as f:
            f.write("# Dropped Rows Summary\n\n")
            
            if not dropped_int_rows.empty:
                f.write(f"## Intersection Data: {len(dropped_int_rows)} rows dropped\n\n")
                f.write("Reasons for dropping:\n")
                drop_counts = dropped_int_rows['drop_reason'].value_counts()
                for reason, count in drop_counts.items():
                    f.write(f"- {reason}: {count} rows\n")
                f.write("\nFull details in `dropped_int_rows.csv`\n\n")
            
            if not dropped_link_rows.empty:
                f.write(f"## Link Data: {len(dropped_link_rows)} rows dropped\n\n")
                f.write("Reasons for dropping:\n")
                drop_counts = dropped_link_rows['drop_reason'].value_counts()
                for reason, count in drop_counts.items():
                    f.write(f"- {reason}: {count} rows\n")
                f.write("\nFull details in `dropped_link_rows.csv`\n")
        
        print(f"  Dropped rows information saved to {reports_tables_path}")
    
    # 2.5 Save Cleaned Data
    print("\nSaving cleaned data...")
    
    # Save as CSV
    df_int.to_csv(os.path.join(data_clean_path, "df_int.csv"), index=False)
    df_link.to_csv(os.path.join(data_clean_path, "df_link.csv"), index=False)
    
    # Try to save to parquet if pyarrow is available
    try:
        import pyarrow
        df_int.to_parquet(os.path.join(data_clean_path, "df_int.parquet"))
        df_link.to_parquet(os.path.join(data_clean_path, "df_link.parquet"))
        print("Data saved in both CSV and Parquet formats")
    except ImportError:
        print("Pyarrow not available. Data saved in CSV format only.")
    
    print(f"Cleaned data saved to {data_clean_path}")
    print(f"  - Intersection data: {df_int.shape} rows")
    print(f"  - Link data: {df_link.shape} rows")
    
    print("\nStep 2 completed successfully!")

if __name__ == "__main__":
    main() 