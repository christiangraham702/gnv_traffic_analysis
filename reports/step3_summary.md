# Step 3: Feature Engineering

## Overview
Step 3 focuses on creating derived features from the cleaned data to better represent the traffic patterns and prepare for predictive modeling. This follows the data cleaning and harmonization performed in Step 2.

## Tasks Completed

### 1. Intersection Data Features
Created the following engineered features:
- `Total_Vol`: sum of all turning-movement counts
- `EW_Vol`: sum of eastbound and westbound volumes
- `NS_Vol`: sum of northbound and southbound volumes
- `EW_to_NS`: ratio of east-west to north-south volumes
- `Green_EW`: average of eastbound green times
- `Green_to_Demand_EW`: ratio of green time to eastbound volume
- `Ped_Load`: sum of pedestrian clearance times for east-west and north-south
- `High_Delay`: binary target variable (1 if Delay_s_veh > 60, 0 otherwise)

### 2. Link Data Features
Created the following engineered features:
- `Peak_Ratio`: ratio of PM to AM peak traffic (PkPM_1314 / PkAM_1314)
- `Direction_Balance`: directional balance of traffic (|DlyD1_1314 - DlyD2_1314|)
- `AM_PM_Diff`: absolute difference between AM and PM peaks
- `AM_Peak_Ratio`: AM peak as a proportion of ADT
- `PM_Peak_Ratio`: PM peak as a proportion of ADT

### 3. Visualizations
Generated the following feature visualizations:
- Distribution of Total Volume at intersections
- Total Volume by Delay Category (boxplot)
- Correlation Matrix of Intersection Features
- Distribution of Peak Ratio (PM/AM) on links

## Output Files
- **Engineered Data:**
  - `data/clean/df_int_features.csv` - Intersection data with engineered features
  - `data/clean/df_link_features.csv` - Link data with engineered features

- **Visualizations:**
  - `reports/figures/total_volume_dist.png` - Distribution of total volumes
  - `reports/figures/delay_vs_volume.png` - Relationship between delay and volume
  - `reports/figures/intersection_correlation.png` - Feature correlation matrix
  - `reports/figures/peak_ratio_dist.png` - Distribution of peak ratios

## Key Findings
- Calculated volume features provide insights into traffic distribution patterns
- Created binary target variable for high delay prediction
- Generated peak ratio features to understand temporal traffic patterns
- Visualized relationships between traffic volumes and delays

## Next Steps
The next step (Step 4) will focus on exploratory data analysis:
1. Analyzing the distributions of key features
2. Investigating relationships between features
3. Building correlation matrices to identify key predictors
4. Visualizing geographic patterns in the data 