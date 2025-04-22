# Step 2: Data Cleaning & Harmonization

## Overview
Step 2 focused on cleaning and harmonizing the traffic datasets to prepare them for feature engineering and analysis. This step follows the data ingestion and schema audit completed in Step 1.

## Tasks Completed

### 1. Type Coercion
- Converted percentage values in heavy-vehicle columns to float format by removing "%" and dividing by 100
- Applied `pd.to_numeric()` with `errors='coerce'` to convert green time and lane count columns to proper numeric types

### 2. Datetime Conversion
- Created a Timestamp column from Period and Time fields
- Used a mapping approach to convert AM/PM periods to months (January for AM, July for PM)
- Added a reference Year (2014) where needed

### 3. Missing Value Handling
- Identified columns with missing values and calculated missing percentage
- Applied imputation for columns with <5% missing values:
  - Used median for numeric columns
  - Used mode (most frequent value) for categorical columns
- Marked rows with critical missing data or high missing percentage for potential removal
- Created detailed reports of dropped rows in reports/tables/

### 4. Data Saving
- Saved cleaned data in CSV format (primary)
- Added fallback support to try using Parquet format when pyarrow is available
- Updated the Streamlit app to handle both formats

## Output Files
- **Cleaned Data:**
  - `data/clean/df_int.csv` - Cleaned intersection data
  - `data/clean/df_link.csv` - Cleaned link data

- **Reports:**
  - `reports/tables/dropped_rows_summary.md` - Summary of dropped rows
  - `reports/tables/dropped_int_rows.csv` - Detailed log of dropped intersection rows
  - `reports/tables/dropped_link_rows.csv` - Detailed log of dropped link rows

## Findings
- Heavy columns were successfully converted to float values
- Many green time and lane count columns had missing values
- Timestamp creation identified time-related issues in the data
- Clean data is ready for feature engineering in Step 3

## Next Steps
The next step (Step 3) will focus on feature engineering:
1. Creating derived features from the cleaned data
2. Calculating traffic volume totals and ratios
3. Engineering delay-related target variables
4. Preparing data for modeling 