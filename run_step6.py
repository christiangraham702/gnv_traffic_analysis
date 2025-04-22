#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6: Scenario Generator & Transfer Learning

This script implements Step 6 of the traffic analysis project:
1. Cluster link segments to identify traffic patterns
2. Generate synthetic scenarios based on cluster patterns
3. Predict delay probabilities for these scenarios to find optimal conditions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define project directories
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_TABLES_DIR = REPORTS_DIR / "tables"

# Create directories if they don't exist
os.makedirs(REPORTS_FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_TABLES_DIR, exist_ok=True)

# Fix: Print the actual path for debugging
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_CLEAN_DIR: {DATA_CLEAN_DIR}")
print(f"Looking for files in: {DATA_CLEAN_DIR}")

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

def load_model(model_path="high_delay_rf.pkl"):
    """Load the trained model"""
    print(f"Loading model from {model_path}...")
    try:
        model = joblib.load(MODELS_DIR / model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
def load_model_features(features_file="model_features.csv"):
    """Load the list of features required by the model"""
    print(f"Loading model features from {features_file}...")
    try:
        features = pd.read_csv(MODELS_DIR / features_file)
        feature_list = features['Feature'].tolist()
        print(f"Loaded {len(feature_list)} features: {feature_list}")
        return feature_list
    except Exception as e:
        print(f"Error loading model features: {e}")
        return None

def prepare_link_data_for_clustering(df_link):
    """Prepare link data for clustering by selecting and scaling features"""
    print("Preparing link data for clustering...")
    
    # Select features for clustering
    clustering_features = ['ADT_1314', 'Peak_Ratio', 'HvyPct_1314']
    
    # Check if we have all the features
    valid_features = [col for col in clustering_features if col in df_link.columns]
    if len(valid_features) < len(clustering_features):
        missing = set(clustering_features) - set(valid_features)
        print(f"Warning: Missing clustering features: {missing}")
        
    # Only keep rows with all valid features
    df_cluster = df_link[valid_features].dropna()
    
    print(f"Data for clustering: {df_cluster.shape} rows with features: {valid_features}")
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_cluster)
    
    return df_cluster, scaled_features, scaler, valid_features

def cluster_link_segments(scaled_features, n_clusters=5):
    """Cluster link segments based on ADT, Peak_Ratio, and Heavy_%"""
    print(f"Clustering link segments into {n_clusters} clusters...")
    
    # Build the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    
    # Get cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    return kmeans, labels, centroids

def visualize_clusters(df_cluster, labels, centroids, features):
    """Create visualizations of the clusters"""
    print("Creating cluster visualizations...")
    
    # Add cluster labels to the dataframe
    df_cluster_viz = df_cluster.copy()
    df_cluster_viz['Cluster'] = labels
    
    # Create a pairplot to visualize clusters
    plt.figure(figsize=(10, 8))
    
    if len(features) >= 2:
        # Create a scatter plot of the first two features
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            df_cluster_viz[features[0]], 
            df_cluster_viz[features[1]],
            c=df_cluster_viz['Cluster'], 
            cmap='viridis', 
            alpha=0.6
        )
        
        # Plot the centroids
        plt.scatter(
            centroids[:, 0], 
            centroids[:, 1], 
            c='red', 
            marker='X', 
            s=200, 
            label='Centroids'
        )
        
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title(f'Clusters of Link Segments ({features[0]} vs {features[1]})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORTS_FIGURES_DIR / "link_clusters.png")
        plt.close()
    
    # Create a table summarizing cluster characteristics
    cluster_summary = df_cluster_viz.groupby('Cluster').mean()
    
    # Save the cluster summary to CSV
    cluster_summary.to_csv(REPORTS_TABLES_DIR / "cluster_summary.csv")
    
    print(f"Cluster visualizations saved to {REPORTS_FIGURES_DIR}")
    print("Cluster summary:")
    print(cluster_summary)
    
    return df_cluster_viz, cluster_summary

def generate_synthetic_scenarios(df_int, df_cluster_viz, cluster_summary, model_features, n_scenarios=5):
    """Generate synthetic scenarios based on cluster patterns"""
    print(f"Generating {n_scenarios} synthetic scenarios per intersection...")
    
    # Get unique intersections
    intersections = df_int['INTERSECT'].unique() if 'INTERSECT' in df_int.columns else ['Unknown']
    
    # Create empty dataframe for synthetic scenarios
    synthetic_scenarios = []
    
    # For each intersection
    for intersection in intersections:
        # Get intersection data
        if 'INTERSECT' in df_int.columns:
            int_data = df_int[df_int['INTERSECT'] == intersection].copy()
        else:
            int_data = df_int.copy()
        
        if len(int_data) == 0:
            continue
            
        # Get average values for the intersection
        avg_total_vol = int_data['Total_Vol'].mean() if 'Total_Vol' in int_data.columns else 0
        avg_ew_to_ns = int_data['EW_to_NS'].mean() if 'EW_to_NS' in int_data.columns else 1
        
        # Find the closest cluster based on volume
        cluster_distances = []
        for idx, row in cluster_summary.iterrows():
            if 'ADT_1314' in row and 'Peak_Ratio' in row:
                # Calculate distance to cluster centroid
                vol_dist = abs(row['ADT_1314'] - avg_total_vol) / max(1, avg_total_vol)
                ratio_dist = abs(row['Peak_Ratio'] - avg_ew_to_ns) / max(1, avg_ew_to_ns)
                
                # Combined distance (weighted)
                distance = vol_dist * 0.7 + ratio_dist * 0.3
                
                cluster_distances.append((idx, distance))
        
        # Sort clusters by distance
        cluster_distances.sort(key=lambda x: x[1])
        
        # Get cluster members for the closest cluster (Top 2 clusters)
        top_clusters = [cd[0] for cd in cluster_distances[:2]]
        relevant_links = df_cluster_viz[df_cluster_viz['Cluster'].isin(top_clusters)]
        
        # Generate synthetic scenarios for this intersection
        for i in range(n_scenarios):
            # Create a base scenario from the intersection's average values
            scenario = {
                'INTERSECT': intersection,
                'Scenario': f"S{i+1}_{intersection}",
                'Cluster_Source': top_clusters[i % len(top_clusters)]
            }
            
            # Vary the traffic volume (reduced by i*10%)
            vol_reduction = 1.0 - (i * 0.1)
            
            # Add all required model features
            for feature in model_features:
                if feature == 'Total_Vol':
                    scenario[feature] = avg_total_vol * vol_reduction
                elif feature == 'EW_Vol':
                    # Adjust based on EW to NS ratio
                    if 'EW_Vol' in int_data.columns and 'NS_Vol' in int_data.columns:
                        avg_ew_vol = int_data['EW_Vol'].mean()
                        scenario[feature] = avg_ew_vol * vol_reduction
                    else:
                        # Default: assume 60% of total volume is EW
                        scenario[feature] = (avg_total_vol * 0.6) * vol_reduction
                elif feature == 'NS_Vol':
                    # Adjust based on EW to NS ratio
                    if 'NS_Vol' in int_data.columns:
                        avg_ns_vol = int_data['NS_Vol'].mean()
                        scenario[feature] = avg_ns_vol * vol_reduction
                    else:
                        # Default: assume 40% of total volume is NS
                        scenario[feature] = (avg_total_vol * 0.4) * vol_reduction
                elif feature == 'EW_to_NS':
                    scenario[feature] = avg_ew_to_ns
                elif feature == 'Green_EW':
                    if 'Green_EW' in int_data.columns:
                        # Increase green time slightly for later scenarios
                        avg_green = int_data['Green_EW'].mean()
                        scenario[feature] = avg_green * (1 + i * 0.05)
                    else:
                        scenario[feature] = 30 # default value
                elif feature == 'Green_to_Demand_EW':
                    if 'Green_to_Demand_EW' in int_data.columns:
                        # This will naturally increase as demand decreases
                        avg_g2d = int_data['Green_to_Demand_EW'].mean()
                        # As volume decreases, this ratio increases
                        scenario[feature] = avg_g2d * (1 + i * 0.1)
                    else:
                        scenario[feature] = 0.1 * (1 + i * 0.1)
                elif feature == 'Ped_Load':
                    if 'Ped_Load' in int_data.columns:
                        # Decrease ped load for some scenarios
                        avg_ped = int_data['Ped_Load'].mean()
                        scenario[feature] = avg_ped * (1 - i * 0.05)
                    else:
                        scenario[feature] = 10 * (1 - i * 0.05)
                elif feature == 'Period_Numeric':
                    # Alternate between periods
                    scenario[feature] = i % 2
                else:
                    # For any other feature, use mean value from intersection data
                    if feature in int_data.columns:
                        scenario[feature] = int_data[feature].mean()
                    else:
                        scenario[feature] = 0
            
            synthetic_scenarios.append(scenario)
    
    # Convert to DataFrame
    df_scenarios = pd.DataFrame(synthetic_scenarios)
    
    # Save scenarios to CSV
    df_scenarios.to_csv(DATA_CLEAN_DIR / "synthetic_scenarios.csv", index=False)
    print(f"Generated {len(df_scenarios)} synthetic scenarios")
    print(f"Saved to {DATA_CLEAN_DIR / 'synthetic_scenarios.csv'}")
    
    return df_scenarios

def predict_delay(df_scenarios, model, model_features):
    """Predict delay probabilities for synthetic scenarios"""
    print("Predicting delay probabilities for synthetic scenarios...")
    
    # Ensure all required features are present
    for feature in model_features:
        if feature not in df_scenarios.columns:
            print(f"Warning: Feature {feature} missing from scenarios. Adding default values.")
            df_scenarios[feature] = 0
    
    # Select only the features needed for prediction
    X_scenarios = df_scenarios[model_features]
    
    # Predict probabilities
    try:
        # Get probability of high delay (class 1)
        y_proba = model.predict_proba(X_scenarios)
        
        # Add probability to the scenarios dataframe
        df_scenarios['High_Delay_Prob'] = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba
        
        # Flag "good" scenarios (low probability of high delay)
        df_scenarios['Good_Scenario'] = (df_scenarios['High_Delay_Prob'] < 0.3).astype(int)
        
        # Save predictions to CSV
        df_scenarios.to_csv(DATA_CLEAN_DIR / "scenario_predictions.csv", index=False)
        print(f"Predictions saved to {DATA_CLEAN_DIR / 'scenario_predictions.csv'}")
        
        # Report on number of good scenarios
        good_count = df_scenarios['Good_Scenario'].sum()
        print(f"Found {good_count} good scenarios ({good_count/len(df_scenarios)*100:.1f}%)")
        
        return df_scenarios
        
    except Exception as e:
        print(f"Error predicting delay: {e}")
        return df_scenarios

def analyze_scenarios(df_scenarios):
    """Analyze the scenarios to identify patterns that lead to lower delay"""
    print("Analyzing scenarios to identify patterns for delay reduction...")
    
    # Create a markdown analysis file
    analysis_file = REPORTS_DIR / "step6_scenario_analysis.md"
    
    # Compare good vs bad scenarios
    df_good = df_scenarios[df_scenarios['Good_Scenario'] == 1]
    df_bad = df_scenarios[df_scenarios['Good_Scenario'] == 0]
    
    # Calculate averages for each group
    good_avg = df_good.mean(numeric_only=True)
    bad_avg = df_bad.mean(numeric_only=True)
    
    # Calculate percent difference
    pct_diff = {}
    for col in good_avg.index:
        if col not in ['Scenario', 'INTERSECT', 'Cluster_Source', 'High_Delay_Prob', 'Good_Scenario']:
            if bad_avg[col] != 0:
                pct_diff[col] = (good_avg[col] - bad_avg[col]) / bad_avg[col] * 100
            else:
                pct_diff[col] = float('nan')
    
    # Create a sorted dataframe of the differences
    diff_df = pd.DataFrame({
        'Good_Scenario_Avg': good_avg,
        'Bad_Scenario_Avg': bad_avg,
        'Pct_Difference': pd.Series(pct_diff)
    })
    
    # Sort by absolute percentage difference
    diff_df = diff_df.sort_values(by='Pct_Difference', key=abs, ascending=False)
    
    # Save to CSV
    diff_df.to_csv(REPORTS_TABLES_DIR / "scenario_comparison.csv")
    
    # Visualize key differences
    plt.figure(figsize=(12, 8))
    
    # Select top features with the biggest differences
    top_features = diff_df.index[:5]
    
    # Plot these features
    feature_data = []
    for feature in top_features:
        if feature not in ['High_Delay_Prob', 'Good_Scenario']:
            feature_data.append({
                'Feature': feature,
                'Good_Scenarios': good_avg[feature],
                'Bad_Scenarios': bad_avg[feature]
            })
    
    feature_df = pd.DataFrame(feature_data)
    
    # Melt for easier plotting
    melted_df = pd.melt(
        feature_df, 
        id_vars=['Feature'], 
        value_vars=['Good_Scenarios', 'Bad_Scenarios'],
        var_name='Scenario_Type', 
        value_name='Value'
    )
    
    # Create grouped bar chart
    sns.barplot(x='Feature', y='Value', hue='Scenario_Type', data=melted_df)
    plt.title('Key Features: Good vs Bad Scenarios')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORTS_FIGURES_DIR / "scenario_comparison.png")
    plt.close()
    
    # Create the markdown summary
    with open(analysis_file, 'w') as f:
        f.write("# Step 6: Scenario Analysis Summary\n\n")
        f.write("## Overview\n")
        f.write("This document summarizes the findings from the scenario generation and analysis process.\n")
        f.write("Synthetic scenarios were created based on link traffic patterns and evaluated for delay probability.\n\n")
        
        f.write("## Scenario Generation\n")
        f.write(f"- Total scenarios generated: {len(df_scenarios)}\n")
        f.write(f"- Good scenarios (delay probability < 0.3): {len(df_good)}\n")
        f.write(f"- Bad scenarios (delay probability ≥ 0.3): {len(df_bad)}\n\n")
        
        f.write("## Key Differences Between Good and Bad Scenarios\n")
        f.write("The following features showed the largest differences between good and bad scenarios:\n\n")
        
        f.write("| Feature | Good Scenario Avg | Bad Scenario Avg | % Difference |\n")
        f.write("|---------|------------------|------------------|-------------|\n")
        
        for feature in top_features:
            if feature not in ['High_Delay_Prob', 'Good_Scenario']:
                f.write(f"| {feature} | {good_avg[feature]:.2f} | {bad_avg[feature]:.2f} | {diff_df.loc[feature, 'Pct_Difference']:.1f}% |\n")
        
        f.write("\n## Recommendations for Delay Reduction\n")
        
        # Add recommendations based on the differences
        for feature in top_features:
            if feature not in ['High_Delay_Prob', 'Good_Scenario']:
                if diff_df.loc[feature, 'Pct_Difference'] > 0:
                    f.write(f"- **Increase {feature}**: Good scenarios have {abs(diff_df.loc[feature, 'Pct_Difference']):.1f}% higher values\n")
                else:
                    f.write(f"- **Decrease {feature}**: Good scenarios have {abs(diff_df.loc[feature, 'Pct_Difference']):.1f}% lower values\n")
        
        f.write("\n## Next Steps\n")
        f.write("These findings will be incorporated into the final interactive dashboard to allow users to:\n")
        f.write("1. Explore the impact of different traffic and signal timing parameters\n")
        f.write("2. Test scenarios for specific intersections\n")
        f.write("3. Identify optimal configurations for minimizing delay\n")
    
    print(f"Scenario analysis saved to {analysis_file}")
    print(f"Feature comparison saved to {REPORTS_TABLES_DIR / 'scenario_comparison.csv'}")
    print(f"Visualization saved to {REPORTS_FIGURES_DIR / 'scenario_comparison.png'}")

def main():
    """Main function to run Step 6: Scenario Generator & Transfer Learning"""
    print("Starting Step 6: Scenario Generator & Transfer Learning...")
    
    # Load data
    df_int, df_link = load_data()
    
    # Load models and features
    model = load_model()
    model_features = load_model_features()
    
    if model is None or model_features is None:
        print("Error: Could not load model or model features. Exiting.")
        return
    
    # 6.1 Cluster link segments
    df_cluster, scaled_features, scaler, valid_features = prepare_link_data_for_clustering(df_link)
    kmeans, labels, centroids = cluster_link_segments(scaled_features)
    df_cluster_viz, cluster_summary = visualize_clusters(df_cluster, labels, centroids, valid_features)
    
    # 6.2 Generate synthetic scenarios
    df_scenarios = generate_synthetic_scenarios(df_int, df_cluster_viz, cluster_summary, model_features)
    
    # 6.3 Predict delay probabilities
    df_scenarios = predict_delay(df_scenarios, model, model_features)
    
    # 6.4 Analyze scenarios
    analyze_scenarios(df_scenarios)
    
    print("\nStep 6 completed successfully!")

if __name__ == "__main__":
    main() 