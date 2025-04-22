#!/usr/bin/env python
"""
Run Step 1: Data Ingestion and Schema Audit
This script executes the data loading and schema auditing steps
without needing to open the Jupyter notebook.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the random seed for reproducibility
np.random.seed(42)

# Set up paths
data_raw_path = "data/raw/"
data_clean_path = "data/clean/"
reports_tables_path = "reports/tables/"

# Make sure directories exist
os.makedirs(reports_tables_path, exist_ok=True)

def main():
    print("Running Step 1: Data Ingestion and Schema Audit")
    
    # 1.1 Load Files
    print("\nLoading data files...")
    df_int = pd.read_csv(os.path.join(data_raw_path, "traffic_data.csv"))
    print(f"Intersection data shape: {df_int.shape}")
    
    df_link = pd.read_csv(os.path.join(data_raw_path, "lots_traffic_data.csv"))
    print(f"Link data shape: {df_link.shape}")
    
    # 1.2-1.3 Schema Audit
    print("\nExporting schema information...")
    with open(os.path.join(reports_tables_path, "raw_schema.md"), "w") as f:
        f.write("# Raw Data Schema Information\n\n")
        
        # Intersection data
        f.write("## Intersection Data (`traffic_data.csv`)\n\n")
        f.write(f"* Shape: {df_int.shape}\n")
        f.write("* Columns:\n\n")
        
        for col in df_int.columns:
            f.write(f"  - `{col}`: {df_int[col].dtype}\n")
        
        # Link data
        f.write("\n## Link Data (`lots_traffic_data.csv`)\n\n")
        f.write(f"* Shape: {df_link.shape}\n")
        f.write("* Columns:\n\n")
        
        for col in df_link.columns:
            f.write(f"  - `{col}`: {df_link[col].dtype}\n")
    
    print(f"Schema information exported to {os.path.join(reports_tables_path, 'raw_schema.md')}")
    
    # 1.5 Assert Uniqueness Keys
    print("\nChecking uniqueness constraints...")
    
    # Check intersection data uniqueness
    int_key_cols = ["INTERSECT", "Period", "Time"]
    int_unique_count = df_int.groupby(int_key_cols).size().reset_index(name='count')
    is_int_unique = len(df_int) == len(int_unique_count)
    
    print(f"Intersection data uniqueness check on {int_key_cols}:")
    print(f"  - Total rows: {len(df_int)}")
    print(f"  - Unique combinations: {len(int_unique_count)}")
    print(f"  - Is unique: {is_int_unique}")
    
    # Check link data uniqueness (assuming Station_ID and Year columns exist)
    link_key_cols = []
    if 'Station_ID' in df_link.columns:
        link_key_cols.append('Station_ID')
    if 'Year' in df_link.columns:
        link_key_cols.append('Year')
    
    if link_key_cols:
        link_unique_count = df_link.groupby(link_key_cols).size().reset_index(name='count')
        is_link_unique = len(df_link) == len(link_unique_count)
        
        print(f"Link data uniqueness check on {link_key_cols}:")
        print(f"  - Total rows: {len(df_link)}")
        print(f"  - Unique combinations: {len(link_unique_count)}")
        print(f"  - Is unique: {is_link_unique}")
    else:
        print("Could not check link data uniqueness. Required columns not found.")
        print(f"Available columns: {df_link.columns.tolist()}")
    
    print("\nStep 1 completed successfully!")

if __name__ == "__main__":
    main() 