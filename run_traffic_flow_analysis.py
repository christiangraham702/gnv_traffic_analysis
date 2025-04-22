#!/usr/bin/env python
"""
Traffic Flow Analysis
---------------------
This script analyzes the link dataset to understand traffic flow patterns,
directional imbalances, and peak hour congestion across the road network.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap
import json

# Set the random seed for reproducibility
np.random.seed(42)

# Set up paths - use absolute paths for reliability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CLEAN_DIR = os.path.join(SCRIPT_DIR, "data", "clean")
REPORTS_FIGURES_DIR = os.path.join(SCRIPT_DIR, "reports", "figures")
REPORTS_TRAFFIC_FLOW_DIR = os.path.join(SCRIPT_DIR, "reports", "traffic_flow")

# Make sure directories exist
os.makedirs(REPORTS_FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_TRAFFIC_FLOW_DIR, exist_ok=True)

def load_link_data():
    """Load the link dataset"""
    try:
        # Try parquet first for best performance
        df_link = pd.read_parquet(os.path.join(DATA_CLEAN_DIR, "df_link_features.parquet"))
        print("Loaded link features data from parquet file")
    except:
        try:
            # Fall back to CSV
            df_link = pd.read_csv(os.path.join(DATA_CLEAN_DIR, "df_link_features.csv"))
            print("Loaded link features data from CSV file")
        except:
            # Try raw data as last resort
            df_link = pd.read_csv(os.path.join(DATA_CLEAN_DIR, "df_link.csv"))
            print("Loaded link data from CSV file")
    
    # Find all ADT columns (from different years)
    adt_columns = [col for col in df_link.columns if 'ADT_' in col]
    print(f"Found {len(adt_columns)} ADT columns: {', '.join(adt_columns)}")
    
    if not adt_columns:
        print("Error: No ADT columns found in the dataset")
        return df_link
    
    # Create a new column that takes the most recent ADT value for each row
    # First sort columns by year (assuming format ADT_YYYY)
    adt_columns_sorted = sorted(adt_columns, key=lambda x: x.split('_')[1] if '_' in x and len(x.split('_')) > 1 else '0000', reverse=True)
    
    # Create a new ADT_Valid column using the most recent available ADT for each road
    df_link['ADT_Valid'] = None
    df_link['ADT_Year'] = None
    
    for col in adt_columns_sorted:
        # For rows that still have null ADT_Valid, fill with this column's value if it's valid
        mask = (df_link['ADT_Valid'].isna()) & (df_link[col].notna()) & (df_link[col] > 0)
        df_link.loc[mask, 'ADT_Valid'] = df_link.loc[mask, col]
        year = col.split('_')[1] if len(col.split('_')) > 1 else "Unknown"
        df_link.loc[mask, 'ADT_Year'] = year
    
    # Convert to numeric to ensure proper filtering
    df_link['ADT_Valid'] = pd.to_numeric(df_link['ADT_Valid'], errors='coerce')
    
    # Filter out rows without any valid ADT
    df_link_filtered = df_link.dropna(subset=['ADT_Valid'])
    df_link_filtered = df_link_filtered[df_link_filtered['ADT_Valid'] > 0]
    
    # Count how many rows were filtered for each ADT year 
    adt_year_counts = df_link_filtered['ADT_Year'].value_counts().to_dict()
    adt_year_str = ", ".join([f"{year}: {count}" for year, count in adt_year_counts.items()])
    
    print(f"Filtered data: {len(df_link_filtered)} rows with valid ADT values")
    print(f"ADT data sources: {adt_year_str}")
    
    # Required columns (except for ADT which we've handled above)
    required_non_adt_cols = ['the_geom', 'STREET', 'BLOCK', 'PkAM_1314', 'PkPM_1314', 'DlyD1_1314', 'DlyD2_1314']
    
    if all(col in df_link_filtered.columns for col in required_non_adt_cols):
        # Filter for rows with complete data for other key analysis columns
        df_link_filtered = df_link_filtered.dropna(subset=required_non_adt_cols)
        print(f"Final filtered data: {len(df_link_filtered)} rows with complete data")
    else:
        missing_cols = [col for col in required_non_adt_cols if col not in df_link_filtered.columns]
        print(f"Warning: Missing columns: {missing_cols}")
    
    return df_link_filtered

def extract_coordinates(df_link):
    """Extract longitude and latitude from the_geom column"""
    if 'the_geom' not in df_link.columns:
        print("Error: the_geom column not found in link data")
        return df_link
    
    # Extract coordinates from POINT format
    # Format example: POINT (-82.32454871717412 29.65991213468399)
    coords = df_link['the_geom'].str.extract(r'POINT \(([^ ]+) ([^)]+)\)')
    
    if coords is not None and not coords.empty:
        df_link['longitude'] = coords[0].astype(float)
        df_link['latitude'] = coords[1].astype(float)
        print("Extracted coordinates from the_geom column")
    else:
        print("Error extracting coordinates from the_geom column")
    
    return df_link

def convert_to_geopandas(df_link):
    """Convert DataFrame to GeoDataFrame with Point geometry"""
    if 'longitude' not in df_link.columns or 'latitude' not in df_link.columns:
        print("Error: longitude and latitude columns not found")
        return None
    
    # Create Point geometry from longitude and latitude
    geometry = [Point(lon, lat) for lon, lat in zip(df_link['longitude'], df_link['latitude'])]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df_link, geometry=geometry, crs="EPSG:4326")
    print(f"Created GeoDataFrame with {len(gdf)} points")
    
    return gdf

def calculate_flow_metrics(df_link):
    """Calculate additional traffic flow metrics for analysis"""
    print("Calculating traffic flow metrics...")
    
    # Create directional imbalance metrics (if not already present)
    if 'Direction_Balance' not in df_link.columns:
        if all(col in df_link.columns for col in ['DlyD1_1314', 'DlyD2_1314']):
            df_link['Direction_Balance'] = abs(df_link['DlyD1_1314'] - df_link['DlyD2_1314'])
            print("  - Created Direction_Balance metric")
    
    # Calculate normalized peak-to-ADT ratios (if not already present)
    if 'AM_Peak_Ratio' not in df_link.columns:
        if all(col in df_link.columns for col in ['PkAM_1314', 'ADT_Valid']):
            df_link['AM_Peak_Ratio'] = df_link['PkAM_1314'] / df_link['ADT_Valid'].replace(0, np.nan)
            print("  - Created AM_Peak_Ratio metric")
    
    if 'PM_Peak_Ratio' not in df_link.columns:
        if all(col in df_link.columns for col in ['PkPM_1314', 'ADT_Valid']):
            df_link['PM_Peak_Ratio'] = df_link['PkPM_1314'] / df_link['ADT_Valid'].replace(0, np.nan)
            print("  - Created PM_Peak_Ratio metric")
    
    # AM/PM dominant direction metrics
    if all(col in df_link.columns for col in ['AMD1_1314', 'AMD2_1314']):
        df_link['AM_Dominant_Direction'] = np.where(df_link['AMD1_1314'] > df_link['AMD2_1314'], 'D1', 'D2')
        print("  - Created AM_Dominant_Direction metric")
    
    if all(col in df_link.columns for col in ['PMD1_1314', 'PMD2_1314']):
        df_link['PM_Dominant_Direction'] = np.where(df_link['PMD1_1314'] > df_link['PMD2_1314'], 'D1', 'D2')
        print("  - Created PM_Dominant_Direction metric")
    
    # Direction switch between AM and PM (commuter flow indicator)
    if all(col in df_link.columns for col in ['AM_Dominant_Direction', 'PM_Dominant_Direction']):
        df_link['Direction_Switch'] = (df_link['AM_Dominant_Direction'] != df_link['PM_Dominant_Direction']).astype(int)
        print("  - Created Direction_Switch indicator")
    
    # Traffic intensity categories
    if 'ADT_Valid' in df_link.columns:
        try:
            # Use simple percentile-based categorization
            adt_values = df_link['ADT_Valid'].dropna()
            
            # Skip if too many identical values
            if len(adt_values.unique()) < 5:
                raise ValueError("Not enough unique values for categorization")
                
            # Create quantiles manually
            q_20 = adt_values.quantile(0.2)
            q_40 = adt_values.quantile(0.4)
            q_60 = adt_values.quantile(0.6)
            q_80 = adt_values.quantile(0.8)
            
            # Create categories
            conditions = [
                (df_link['ADT_Valid'] <= q_20),
                (df_link['ADT_Valid'] > q_20) & (df_link['ADT_Valid'] <= q_40),
                (df_link['ADT_Valid'] > q_40) & (df_link['ADT_Valid'] <= q_60),
                (df_link['ADT_Valid'] > q_60) & (df_link['ADT_Valid'] <= q_80),
                (df_link['ADT_Valid'] > q_80)
            ]
            
            # Apply categories
            categories = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            df_link['Traffic_Intensity'] = np.select(conditions, categories, default=np.nan)
            print("  - Created Traffic_Intensity categories")
        except Exception as e:
            print(f"  - Could not create Traffic_Intensity categories: {e}")
            # Create a simpler categorical version instead
            df_link['Traffic_Intensity'] = np.where(df_link['ADT_Valid'] > df_link['ADT_Valid'].median(), 
                                                  'High', 'Low')
            print("  - Created simple Traffic_Intensity categories (High/Low)")
    
    # Peak hour congestion indicator
    if all(col in df_link.columns for col in ['AM_Peak_Ratio', 'PM_Peak_Ratio']):
        try:
            # Calculate max peak ratio
            df_link['Max_Peak_Ratio'] = df_link[['AM_Peak_Ratio', 'PM_Peak_Ratio']].max(axis=1)
            max_values = df_link['Max_Peak_Ratio'].dropna()
            
            # Skip if too many identical values
            if len(max_values.unique()) < 5:
                raise ValueError("Not enough unique values for categorization")
                
            # Create quantiles manually
            q_20 = max_values.quantile(0.2)
            q_40 = max_values.quantile(0.4)
            q_60 = max_values.quantile(0.6)
            q_80 = max_values.quantile(0.8)
            
            # Create categories
            conditions = [
                (df_link['Max_Peak_Ratio'] <= q_20),
                (df_link['Max_Peak_Ratio'] > q_20) & (df_link['Max_Peak_Ratio'] <= q_40),
                (df_link['Max_Peak_Ratio'] > q_40) & (df_link['Max_Peak_Ratio'] <= q_60),
                (df_link['Max_Peak_Ratio'] > q_60) & (df_link['Max_Peak_Ratio'] <= q_80),
                (df_link['Max_Peak_Ratio'] > q_80)
            ]
            
            # Apply categories
            categories = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            df_link['Peak_Congestion'] = np.select(conditions, categories, default=np.nan)
            print("  - Created Peak_Congestion indicator")
        except Exception as e:
            print(f"  - Could not create Peak_Congestion indicator: {e}")
            # Create a simpler version
            if 'Max_Peak_Ratio' not in df_link.columns:
                df_link['Max_Peak_Ratio'] = df_link[['AM_Peak_Ratio', 'PM_Peak_Ratio']].max(axis=1)
            
            df_link['Peak_Congestion'] = np.where(df_link['Max_Peak_Ratio'] > df_link['Max_Peak_Ratio'].median(), 
                                                'High', 'Low')
            print("  - Created simple Peak_Congestion indicator (High/Low)")
    
    return df_link

def create_correlation_heatmap(df_link):
    """Create a correlation heatmap for traffic flow metrics"""
    print("Creating correlation heatmap for traffic flow metrics...")
    
    # Select numeric columns for correlation analysis
    flow_cols = [
        'ADT_Valid', 'PkAM_1314', 'PkPM_1314', 
        'DlyD1_1314', 'DlyD2_1314', 
        'Direction_Balance', 'Peak_Ratio',
        'AM_Peak_Ratio', 'PM_Peak_Ratio'
    ]
    
    flow_cols = [col for col in flow_cols if col in df_link.columns]
    
    if len(flow_cols) < 2:
        print("Error: Not enough flow metrics for correlation analysis")
        return None
    
    # Calculate correlation
    corr = df_link[flow_cols].corr().round(2)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, 
        annot=True, 
        cmap='coolwarm', 
        mask=mask,
        vmin=-1, vmax=1, 
        fmt='.2f',
        linewidths=0.5
    )
    plt.title('Correlation of Traffic Flow Metrics', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(REPORTS_FIGURES_DIR, "traffic_flow_correlation.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to {output_file}")
    
    return output_file

def visualize_traffic_distributions(df_link):
    """Create visualizations for traffic distribution patterns"""
    print("Creating traffic distribution visualizations...")
    
    # Create directory for visualizations
    os.makedirs(os.path.join(REPORTS_FIGURES_DIR, "traffic_flow"), exist_ok=True)
    
    output_files = []
    
    # 1. AM vs PM Peak Volume Distribution
    if all(col in df_link.columns for col in ['PkAM_1314', 'PkPM_1314']):
        plt.figure(figsize=(10, 6))
        
        # Create a scatter plot
        plt.scatter(
            df_link['PkAM_1314'],
            df_link['PkPM_1314'],
            alpha=0.5,
            s=df_link['ADT_Valid'] / 100 if 'ADT_Valid' in df_link.columns else 20,
            c=df_link['Direction_Balance'] if 'Direction_Balance' in df_link.columns else 'blue'
        )
        
        # Add reference line (x=y)
        max_val = max(df_link['PkAM_1314'].max(), df_link['PkPM_1314'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
        
        # Add labels and title
        plt.xlabel('AM Peak Volume', fontsize=12)
        plt.ylabel('PM Peak Volume', fontsize=12)
        plt.title('AM vs PM Peak Traffic Volume', fontsize=14)
        
        # Add color bar if Direction_Balance is used for color
        if 'Direction_Balance' in df_link.columns:
            cbar = plt.colorbar()
            cbar.set_label('Direction Imbalance')
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(REPORTS_FIGURES_DIR, "traffic_flow", "am_vs_pm_peak_volume.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Saved AM vs PM peak volume plot to {output_file}")
        output_files.append(output_file)
    
    # 2. Direction Balance Distribution
    if 'Direction_Balance' in df_link.columns:
        plt.figure(figsize=(10, 6))
        
        # Create histogram with KDE
        sns.histplot(
            data=df_link,
            x='Direction_Balance',
            kde=True,
            bins=30
        )
        
        # Add vertical line at median
        median = df_link['Direction_Balance'].median()
        plt.axvline(median, color='red', linestyle='--', alpha=0.7, 
                   label=f'Median: {median:.2f}')
        
        # Add labels and title
        plt.xlabel('Direction Balance (|D1-D2|)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Directional Imbalance', fontsize=14)
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(REPORTS_FIGURES_DIR, "traffic_flow", "direction_balance_distribution.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Saved direction balance distribution to {output_file}")
        output_files.append(output_file)
    
    # 3. Peak Ratio Distribution by Traffic Intensity
    if all(col in df_link.columns for col in ['Peak_Ratio', 'Traffic_Intensity']):
        plt.figure(figsize=(12, 6))
        
        # Create boxplot
        sns.boxplot(
            data=df_link,
            x='Traffic_Intensity',
            y='Peak_Ratio'
        )
        
        # Add horizontal line at 1.0 (equal AM/PM)
        plt.axhline(1.0, color='red', linestyle='--', alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Traffic Intensity', fontsize=12)
        plt.ylabel('Peak Ratio (PM/AM)', fontsize=12)
        plt.title('Peak Ratio Distribution by Traffic Intensity', fontsize=14)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(REPORTS_FIGURES_DIR, "traffic_flow", "peak_ratio_by_traffic_intensity.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Saved peak ratio by traffic intensity to {output_file}")
        output_files.append(output_file)
    
    # 4. Direction Switch Analysis (AM vs PM dominant direction)
    if 'Direction_Switch' in df_link.columns:
        plt.figure(figsize=(10, 6))
        
        # Count of links with and without direction switch
        switch_counts = df_link['Direction_Switch'].value_counts()
        
        # Create pie chart
        plt.pie(
            switch_counts,
            labels=['Same Dominant Direction', 'Direction Switch (AM to PM)'],
            autopct='%1.1f%%',
            colors=['#66b3ff', '#ff9999'],
            explode=(0, 0.1)
        )
        
        plt.title('AM to PM Dominant Direction Switch Analysis', fontsize=14)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(REPORTS_FIGURES_DIR, "traffic_flow", "direction_switch_analysis.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Saved direction switch analysis to {output_file}")
        output_files.append(output_file)
    
    return output_files

def create_interactive_map(gdf_link):
    """Create an interactive map for traffic flow analysis"""
    print("Creating interactive traffic flow maps...")
    
    # Check if input is a GeoDataFrame with geometry
    if gdf_link is None or not isinstance(gdf_link, gpd.GeoDataFrame) or not hasattr(gdf_link, 'geometry'):
        print("Error: Valid GeoDataFrame required for map creation")
        return []
    
    # Ensure all geometry entries are valid
    invalid_geom = gdf_link[gdf_link.geometry.isna()].index
    if len(invalid_geom) > 0:
        print(f"Warning: {len(invalid_geom)} rows with invalid geometry found. Removing them for map creation.")
        gdf_link = gdf_link.loc[~gdf_link.index.isin(invalid_geom)].copy()
    
    # Check if we still have valid data after filtering
    if len(gdf_link) == 0:
        print("Error: No valid geometry data available for map creation")
        return []
    
    output_files = []
    
    # Calculate center of the map
    center_lat = gdf_link.geometry.y.mean()
    center_lon = gdf_link.geometry.x.mean()
    
    # 1. Direction Balance Map
    if 'Direction_Balance' in gdf_link.columns:
        # Create map
        m_direction = folium.Map(location=[center_lat, center_lon], zoom_start=13)
        
        # Add title to map
        title_html = '''
        <h3 align="center" style="font-size:16px"><b>Traffic Direction Balance Map</b></h3>
        '''
        m_direction.get_root().html.add_child(folium.Element(title_html))
        
        # Normalize direction balance for color scale
        max_imbalance = gdf_link['Direction_Balance'].max()
        
        # Add points to map
        for idx, row in gdf_link.iterrows():
            # Skip rows with missing coordinates
            if pd.isna(row.geometry.x) or pd.isna(row.geometry.y):
                continue
            
            # Normalize imbalance value for color (0-1 scale)
            imbalance = row['Direction_Balance'] / max_imbalance if max_imbalance > 0 else 0
            
            # Create color scale from green (balanced) to red (imbalanced)
            # Higher imbalance = redder color
            color = f'#{int(255 * imbalance):02x}{int(255 * (1-imbalance)):02x}00'
            
            # Create popup info
            popup_text = f"""
            <b>Street:</b> {row['STREET']}<br>
            <b>Block:</b> {row['BLOCK']}<br>
            <b>Direction Balance:</b> {row['Direction_Balance']:.3f}<br>
            <b>ADT:</b> {row['ADT_Valid']:.0f}<br>
            <b>ADT Year:</b> {row.get('ADT_Year', 'Unknown')}<br>
            <b>AM Peak:</b> {row['PkAM_1314']}<br>
            <b>PM Peak:</b> {row['PkPM_1314']}<br>
            """
            
            # Add marker
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5 + (row['ADT_Valid'] / gdf_link['ADT_Valid'].max() * 10 if 'ADT_Valid' in gdf_link.columns else 5),
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m_direction)
        
        # Add legend
        colormap = folium.LinearColormap(
            ['green', 'yellow', 'red'],
            vmin=0, vmax=max_imbalance,
            caption='Direction Balance (|D1-D2|)'
        )
        colormap.add_to(m_direction)
        
        # Save map
        output_file = os.path.join(REPORTS_TRAFFIC_FLOW_DIR, "direction_balance_map.html")
        m_direction.save(output_file)
        print(f"  - Saved direction balance map to {output_file}")
        output_files.append(output_file)
    
    # 2. AM/PM Peak Traffic Map
    if all(col in gdf_link.columns for col in ['AM_Peak_Ratio', 'PM_Peak_Ratio']):
        try:
            # Create AM map
            m_am = folium.Map(location=[center_lat, center_lon], zoom_start=13)
            
            # Add title to map
            title_html = '''
            <h3 align="center" style="font-size:16px"><b>AM Peak Traffic Heatmap</b></h3>
            '''
            m_am.get_root().html.add_child(folium.Element(title_html))
            
            # Create heatmap data - ensure no NaN values
            heat_data_am = []
            for idx, row in gdf_link.iterrows():
                if (not pd.isna(row.geometry.y) and 
                    not pd.isna(row.geometry.x) and 
                    not pd.isna(row['AM_Peak_Ratio'])):
                    heat_data_am.append([
                        float(row.geometry.y), 
                        float(row.geometry.x), 
                        float(row['AM_Peak_Ratio'] * 100)
                    ])
            
            # Add heatmap layer with fixed gradient values (strings as keys)
            if heat_data_am:
                gradient = {'0.4': 'blue', '0.65': 'lime', '0.8': 'yellow', '1': 'red'}
                HeatMap(heat_data_am, radius=15, gradient=gradient).add_to(m_am)
                
                # Save AM map
                output_file_am = os.path.join(REPORTS_TRAFFIC_FLOW_DIR, "am_peak_heatmap.html")
                m_am.save(output_file_am)
                print(f"  - Saved AM peak heatmap to {output_file_am}")
                output_files.append(output_file_am)
            else:
                print("  - Could not create AM peak heatmap (insufficient data)")
            
            # Create PM map
            m_pm = folium.Map(location=[center_lat, center_lon], zoom_start=13)
            
            # Add title to map
            title_html = '''
            <h3 align="center" style="font-size:16px"><b>PM Peak Traffic Heatmap</b></h3>
            '''
            m_pm.get_root().html.add_child(folium.Element(title_html))
            
            # Create heatmap data - ensure no NaN values
            heat_data_pm = []
            for idx, row in gdf_link.iterrows():
                if (not pd.isna(row.geometry.y) and 
                    not pd.isna(row.geometry.x) and 
                    not pd.isna(row['PM_Peak_Ratio'])):
                    heat_data_pm.append([
                        float(row.geometry.y), 
                        float(row.geometry.x), 
                        float(row['PM_Peak_Ratio'] * 100)
                    ])
            
            # Add heatmap layer with fixed gradient values (strings as keys)
            if heat_data_pm:
                gradient = {'0.4': 'blue', '0.65': 'lime', '0.8': 'yellow', '1': 'red'}
                HeatMap(heat_data_pm, radius=15, gradient=gradient).add_to(m_pm)
                
                # Save PM map
                output_file_pm = os.path.join(REPORTS_TRAFFIC_FLOW_DIR, "pm_peak_heatmap.html")
                m_pm.save(output_file_pm)
                print(f"  - Saved PM peak heatmap to {output_file_pm}")
                output_files.append(output_file_pm)
            else:
                print("  - Could not create PM peak heatmap (insufficient data)")
        except Exception as e:
            print(f"  - Error creating peak heatmaps: {e}")
    
    # 3. ADT Heatmap (showing overall traffic intensity)
    if 'ADT_Valid' in gdf_link.columns:
        try:
            # Create ADT map
            m_adt = folium.Map(location=[center_lat, center_lon], zoom_start=13)
            
            # Add title to map
            title_html = '''
            <h3 align="center" style="font-size:16px"><b>Average Daily Traffic Heatmap</b></h3>
            '''
            m_adt.get_root().html.add_child(folium.Element(title_html))
            
            # Create heatmap data - ensure no NaN values
            heat_data_adt = []
            for idx, row in gdf_link.iterrows():
                if (not pd.isna(row.geometry.y) and 
                    not pd.isna(row.geometry.x) and 
                    not pd.isna(row['ADT_Valid'])):
                    heat_data_adt.append([
                        float(row.geometry.y), 
                        float(row.geometry.x), 
                        float(row['ADT_Valid'] / 100)  # Scale down for better visualization
                    ])
            
            # Add heatmap layer with fixed gradient values
            if heat_data_adt:
                gradient = {'0.4': 'blue', '0.65': 'lime', '0.8': 'yellow', '1': 'red'}
                HeatMap(heat_data_adt, radius=15, gradient=gradient).add_to(m_adt)
                
                # Save ADT map
                output_file_adt = os.path.join(REPORTS_TRAFFIC_FLOW_DIR, "adt_heatmap.html")
                m_adt.save(output_file_adt)
                print(f"  - Saved ADT heatmap to {output_file_adt}")
                output_files.append(output_file_adt)
            else:
                print("  - Could not create ADT heatmap (insufficient data)")
        except Exception as e:
            print(f"  - Error creating ADT heatmap: {e}")
    
    # 4. Direction Switch Map (AM/PM dominant direction change)
    if 'Direction_Switch' in gdf_link.columns:
        try:
            # Create map
            m_switch = folium.Map(location=[center_lat, center_lon], zoom_start=13)
            
            # Add title to map
            title_html = '''
            <h3 align="center" style="font-size:16px"><b>AM/PM Direction Switch Map</b></h3>
            '''
            m_switch.get_root().html.add_child(folium.Element(title_html))
            
            # Add points to map
            for idx, row in gdf_link.iterrows():
                # Skip rows with missing coordinates
                if pd.isna(row.geometry.x) or pd.isna(row.geometry.y):
                    continue
                
                # Choose color based on direction switch
                color = 'red' if row['Direction_Switch'] == 1 else 'blue'
                
                # Create popup info
                popup_text = f"""
                <b>Street:</b> {row['STREET']}<br>
                <b>Block:</b> {row['BLOCK']}<br>
                <b>Direction Switch:</b> {'Yes' if row['Direction_Switch'] == 1 else 'No'}<br>
                <b>AM Dominant:</b> {row.get('AM_Dominant_Direction', 'N/A')}<br>
                <b>PM Dominant:</b> {row.get('PM_Dominant_Direction', 'N/A')}<br>
                <b>ADT:</b> {row['ADT_Valid']:.0f}<br>
                <b>ADT Year:</b> {row.get('ADT_Year', 'Unknown')}<br>
                """
                
                # Add marker
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_text, max_width=300)
                ).add_to(m_switch)
            
            # Add legend (manually)
            legend_html = '''
            <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: 80px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                ">
                <span style="color:blue;">&#9679;</span> Same dominant direction<br>
                <span style="color:red;">&#9679;</span> Direction switches AM/PM
            </div>
            '''
            m_switch.get_root().html.add_child(folium.Element(legend_html))
            
            # Save map
            output_file = os.path.join(REPORTS_TRAFFIC_FLOW_DIR, "direction_switch_map.html")
            m_switch.save(output_file)
            print(f"  - Saved direction switch map to {output_file}")
            output_files.append(output_file)
        except Exception as e:
            print(f"  - Error creating direction switch map: {e}")
    
    return output_files

def export_data_for_dashboard(df_link, gdf_link=None):
    """Export processed data for the dashboard"""
    print("Exporting processed data for dashboard...")
    
    # Create a copy with only necessary columns for the dashboard
    dashboard_cols = [
        'STREET', 'BLOCK', 'ADT_Valid', 'PkAM_1314', 'PkPM_1314',
        'DlyD1_1314', 'DlyD2_1314', 'Heavy_1314',
        # Engineered features
        'Peak_Ratio', 'Direction_Balance', 'AM_PM_Diff',
        'AM_Peak_Ratio', 'PM_Peak_Ratio',
        # New metrics
        'AM_Dominant_Direction', 'PM_Dominant_Direction', 'Direction_Switch',
        'Traffic_Intensity', 'Peak_Congestion',
        # Coordinates
        'longitude', 'latitude'
    ]
    
    # Filter to columns that exist
    existing_cols = [col for col in dashboard_cols if col in df_link.columns]
    df_dashboard = df_link[existing_cols].copy()
    
    # Save to parquet for efficient loading
    output_file = os.path.join(DATA_CLEAN_DIR, "traffic_flow_data.parquet")
    df_dashboard.to_parquet(output_file, index=False)
    print(f"Saved dashboard data to {output_file}")
    
    # Export GeoJSON for maps if GeoDataFrame is available
    if gdf_link is not None and isinstance(gdf_link, gpd.GeoDataFrame):
        try:
            # Simplify to key columns for GeoJSON
            map_cols = [
                'STREET', 'BLOCK', 'ADT_Valid', 'PkAM_1314', 'PkPM_1314',
                'Direction_Balance', 'Peak_Ratio', 
                'AM_Peak_Ratio', 'PM_Peak_Ratio',
                'Direction_Switch'
            ]
            
            # Get geometry column name to ensure it's included
            geometry_col = gdf_link._geometry_column_name
            
            # Filter to columns that exist and ensure we keep geometry
            existing_map_cols = [col for col in map_cols if col in gdf_link.columns]
            if geometry_col not in existing_map_cols:
                existing_map_cols.append(geometry_col)
                
            # Create GeoDataFrame subset with selected columns
            gdf_map = gpd.GeoDataFrame(
                gdf_link[existing_map_cols],
                geometry=gdf_link.geometry,
                crs=gdf_link.crs
            )
            
            # Save to GeoJSON
            output_geojson = os.path.join(DATA_CLEAN_DIR, "traffic_flow_map.geojson")
            gdf_map.to_file(output_geojson, driver='GeoJSON')
            print(f"Saved GeoJSON map data to {output_geojson}")
        except Exception as e:
            print(f"Error exporting GeoJSON: {e}")
    else:
        print("GeoDataFrame not available, skipping GeoJSON export")
    
    return output_file

def export_metrics_summary(df_link):
    """Export summary metrics for the dashboard"""
    print("Exporting summary metrics...")
    
    # Helper function to convert NumPy types to native Python types
    def convert_to_python_type(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Calculate summary statistics
    summary = {
        "num_links": int(len(df_link)),
        "num_streets": int(df_link['STREET'].nunique()),
        "total_adt": float(df_link['ADT_Valid'].sum()) if 'ADT_Valid' in df_link.columns else None,
        "avg_am_peak": float(df_link['PkAM_1314'].mean()) if 'PkAM_1314' in df_link.columns else None,
        "avg_pm_peak": float(df_link['PkPM_1314'].mean()) if 'PkPM_1314' in df_link.columns else None,
        "avg_peak_ratio": float(df_link['Peak_Ratio'].mean()) if 'Peak_Ratio' in df_link.columns else None,
        "avg_direction_balance": float(df_link['Direction_Balance'].mean()) if 'Direction_Balance' in df_link.columns else None,
        "pct_direction_switch": float(df_link['Direction_Switch'].mean() * 100) if 'Direction_Switch' in df_link.columns else None,
    }
    
    # Calculate top congested streets
    if 'STREET' in df_link.columns and 'ADT_Valid' in df_link.columns:
        top_streets_series = df_link.groupby('STREET')['ADT_Valid'].mean().sort_values(ascending=False).head(10)
        # Convert to regular Python dict with native types
        top_streets = {str(k): float(v) for k, v in top_streets_series.to_dict().items()}
        summary["top_streets"] = top_streets
    
    # Calculate highest directional imbalance streets
    if 'STREET' in df_link.columns and 'Direction_Balance' in df_link.columns:
        top_imbalanced_series = df_link.groupby('STREET')['Direction_Balance'].mean().sort_values(ascending=False).head(10)
        # Convert to regular Python dict with native types
        top_imbalanced = {str(k): float(v) for k, v in top_imbalanced_series.to_dict().items()}
        summary["top_imbalanced_streets"] = top_imbalanced
    
    try:
        # Save to JSON
        output_file = os.path.join(REPORTS_TRAFFIC_FLOW_DIR, "summary_metrics.json")
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved summary metrics to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error saving summary metrics: {e}")
        return None

def analyze_traffic_flow():
    """Analyze traffic flow data and generate reports"""
    print("Starting traffic flow analysis...")
    
    # Create output directory
    os.makedirs(REPORTS_TRAFFIC_FLOW_DIR, exist_ok=True)
    
    # Load traffic link data
    df_link = load_link_data()
    
    if df_link is None or df_link.empty:
        print("Error: No traffic link data available for analysis")
        return
    
    # Extract coordinates and convert to GeoDataFrame
    df_link = extract_coordinates(df_link)
    gdf_link = convert_to_geopandas(df_link)
    
    # If conversion to GeoDataFrame failed, continue with DataFrame but skip map creation
    if gdf_link is None:
        gdf_link = df_link
        print("Warning: Could not create GeoDataFrame, will skip map creation")
    
    # 1. Calculate summary metrics
    print("Calculating summary metrics...")
    metrics = {
        "total_roads_analyzed": len(gdf_link),
        "total_streets": gdf_link['STREET'].nunique(),
        "avg_adt": float(gdf_link['ADT_Valid'].mean()) if 'ADT_Valid' in gdf_link.columns else None,
        "total_adt": float(gdf_link['ADT_Valid'].sum()) if 'ADT_Valid' in gdf_link.columns else None,
        "adt_by_year": {}
    }
    
    # Count number of each ADT year
    if 'ADT_Year' in gdf_link.columns:
        year_counts = gdf_link['ADT_Year'].value_counts().to_dict()
        metrics["adt_by_year"] = {str(year): count for year, count in year_counts.items()}
    
    # Calculate traffic intensity distribution
    if 'ADT_Valid' in gdf_link.columns:
        # Define traffic intensity categories
        intensity_bins = [0, 2000, 5000, 10000, 20000, float('inf')]
        intensity_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        
        # Add intensity category column
        gdf_link['Traffic_Intensity'] = pd.cut(
            gdf_link['ADT_Valid'], 
            bins=intensity_bins, 
            labels=intensity_labels
        )
        
        # Calculate intensity distribution
        intensity_distribution = gdf_link['Traffic_Intensity'].value_counts().to_dict()
        metrics["traffic_intensity_distribution"] = {str(k): int(v) for k, v in intensity_distribution.items()}
    
    # 2. Calculate direction balance metrics (difference between D1 and D2)
    print("Calculating directional flow metrics...")
    # Check if directional data is available
    has_am_directions = all(col in gdf_link.columns for col in ['PkAM_D1_1314', 'PkAM_D2_1314'])
    has_pm_directions = all(col in gdf_link.columns for col in ['PkPM_D1_1314', 'PkPM_D2_1314']) 

    if has_am_directions and has_pm_directions:
        # Handle potential NaN values
        gdf_link['PkAM_D1_1314'].fillna(0, inplace=True)
        gdf_link['PkAM_D2_1314'].fillna(0, inplace=True)
        gdf_link['PkPM_D1_1314'].fillna(0, inplace=True)
        gdf_link['PkPM_D2_1314'].fillna(0, inplace=True)
        
        # Calculate direction balance for AM and PM
        gdf_link['AM_Direction_Balance'] = abs(gdf_link['PkAM_D1_1314'] - gdf_link['PkAM_D2_1314']) / (
            gdf_link['PkAM_D1_1314'] + gdf_link['PkAM_D2_1314']).replace(0, 1)
        gdf_link['PM_Direction_Balance'] = abs(gdf_link['PkPM_D1_1314'] - gdf_link['PkPM_D2_1314']) / (
            gdf_link['PkPM_D1_1314'] + gdf_link['PkPM_D2_1314']).replace(0, 1)
        
        # Calculate overall direction balance (average of AM and PM)
        gdf_link['Direction_Balance'] = (gdf_link['AM_Direction_Balance'] + gdf_link['PM_Direction_Balance']) / 2
        
        # Calculate dominant direction for AM and PM
        gdf_link['AM_Dominant_Direction'] = np.where(
            gdf_link['PkAM_D1_1314'] > gdf_link['PkAM_D2_1314'], 'D1', 'D2')
        gdf_link['PM_Dominant_Direction'] = np.where(
            gdf_link['PkPM_D1_1314'] > gdf_link['PkPM_D2_1314'], 'D1', 'D2')
        
        # Calculate if there's a switch in dominant direction between AM and PM
        gdf_link['Direction_Switch'] = np.where(
            gdf_link['AM_Dominant_Direction'] != gdf_link['PM_Dominant_Direction'], 1, 0)
        
        # Calculate overall balanced flow percentage
        balanced_count = len(gdf_link[gdf_link['Direction_Balance'] < 0.3])
        balanced_percentage = (balanced_count / len(gdf_link)) * 100 if len(gdf_link) > 0 else 0
        metrics["balanced_flow_percentage"] = balanced_percentage
        metrics["direction_switch_count"] = int(gdf_link['Direction_Switch'].sum())
        metrics["direction_switch_percentage"] = (metrics["direction_switch_count"] / len(gdf_link)) * 100 if len(gdf_link) > 0 else 0
    else:
        # Create simpler directional metrics if we don't have the detailed columns
        if 'DlyD1_1314' in gdf_link.columns and 'DlyD2_1314' in gdf_link.columns:
            # Create a simple direction balance measure using daily values
            gdf_link['Direction_Balance'] = abs(gdf_link['DlyD1_1314'] - gdf_link['DlyD2_1314']) / (
                gdf_link['DlyD1_1314'] + gdf_link['DlyD2_1314']).replace(0, 1)
            
            # Dominant direction based on daily totals
            gdf_link['Dominant_Direction'] = np.where(
                gdf_link['DlyD1_1314'] > gdf_link['DlyD2_1314'], 'D1', 'D2')
            
            # Calculate balanced flow percentage with simplified metric
            balanced_count = len(gdf_link[gdf_link['Direction_Balance'] < 0.3])
            balanced_percentage = (balanced_count / len(gdf_link)) * 100 if len(gdf_link) > 0 else 0
            metrics["balanced_flow_percentage"] = balanced_percentage
            
            # No AM/PM direction switch information available with simplified metrics
            metrics["direction_switch_count"] = 0
            metrics["direction_switch_percentage"] = 0
        else:
            # If we don't have any directional data, add a placeholder
            gdf_link['Direction_Balance'] = 0.5  # Neutral value
            metrics["balanced_flow_percentage"] = 0
            metrics["direction_switch_count"] = 0
            metrics["direction_switch_percentage"] = 0
    
    # 3. Calculate Peak Flow Metrics
    if all(col in gdf_link.columns for col in ['PkAM_1314', 'PkPM_1314', 'ADT_Valid']):
        # Calculate AM and PM peak ratios
        gdf_link['AM_Peak_Ratio'] = gdf_link['PkAM_1314'] / gdf_link['ADT_Valid'].replace(0, np.nan)
        gdf_link['PM_Peak_Ratio'] = gdf_link['PkPM_1314'] / gdf_link['ADT_Valid'].replace(0, np.nan)
        
        # Calculate average peak ratios
        metrics["avg_am_peak_ratio"] = float(gdf_link['AM_Peak_Ratio'].mean())
        metrics["avg_pm_peak_ratio"] = float(gdf_link['PM_Peak_Ratio'].mean())
        
        # Calculate AM vs PM distribution
        am_higher = len(gdf_link[gdf_link['PkAM_1314'] > gdf_link['PkPM_1314']])
        pm_higher = len(gdf_link[gdf_link['PkPM_1314'] > gdf_link['PkAM_1314']])
        metrics["am_higher_percentage"] = (am_higher / len(gdf_link)) * 100 if len(gdf_link) > 0 else 0
        metrics["pm_higher_percentage"] = (pm_higher / len(gdf_link)) * 100 if len(gdf_link) > 0 else 0
    
    # 4. Generate visualizations
    print("Generating traffic flow visualizations...")
    # Create figures directory
    figures_dir = os.path.join(REPORTS_FIGURES_DIR, "traffic_flow")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Generate ADT distribution by year visualization
    if 'ADT_Year' in gdf_link.columns:
        plt.figure(figsize=(10, 6))
        year_counts = gdf_link['ADT_Year'].value_counts().sort_index()
        year_counts.plot(kind='bar', color='skyblue')
        plt.title('ADT Data Distribution by Year')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "adt_by_year.png"))
        plt.close()
    
    # Generate ADT histogram
    if 'ADT_Valid' in gdf_link.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(gdf_link['ADT_Valid'].dropna(), bins=20, color='skyblue', edgecolor='black')
        plt.title('ADT Value Distribution')
        plt.xlabel('ADT Value')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "adt_histogram.png"))
        plt.close()
        
        # Generate ADT box plot by year if year data is available
        if 'ADT_Year' in gdf_link.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='ADT_Year', y='ADT_Valid', data=gdf_link)
            plt.title('ADT Distribution by Year')
            plt.xlabel('Year')
            plt.ylabel('ADT Value')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "adt_by_year_boxplot.png"))
            plt.close()
    
    # Generate Traffic Intensity Pie Chart
    if 'Traffic_Intensity' in gdf_link.columns:
        plt.figure(figsize=(10, 6))
        intensity_counts = gdf_link['Traffic_Intensity'].value_counts()
        intensity_counts.plot(kind='pie', autopct='%1.1f%%', colors=plt.cm.Spectral(np.linspace(0, 1, len(intensity_counts))))
        plt.title('Traffic Intensity Distribution')
        plt.ylabel('')  # Hide the ylabel
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "traffic_intensity_pie.png"))
        plt.close()
    
    # Generate Direction Balance Distribution
    if 'Direction_Balance' in gdf_link.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(gdf_link['Direction_Balance'].dropna(), bins=20, color='skyblue', edgecolor='black')
        plt.title('Direction Balance Distribution')
        plt.xlabel('Direction Balance (0=balanced, 1=one-way)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "direction_balance_hist.png"))
        plt.close()
    
    # Generate Peak Volume Distribution
    if all(col in gdf_link.columns for col in ['PkAM_1314', 'PkPM_1314']):
        # Check which columns exist for directional AM peak volumes
        am_direction_cols = [col for col in ['PkAM_D1_1314', 'PkAM_D2_1314'] if col in gdf_link.columns]
        pm_direction_cols = [col for col in ['PkPM_D1_1314', 'PkPM_D2_1314'] if col in gdf_link.columns]
        
        # Only create heatmaps if we have the directional columns
        if am_direction_cols:
            # AM Peak Volume Heatmap
            plt.figure(figsize=(12, 8))
            pivot_am = gdf_link.pivot_table(
                values=am_direction_cols,
                index='STREET',
                aggfunc='mean'
            ).head(20)  # Top 20 streets for visibility
            
            sns.heatmap(pivot_am, cmap='YlOrRd', annot=True, fmt='.0f')
            plt.title('AM Peak Volume by Direction (Top 20 Streets)')
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "am_peak_volume_heatmap.png"))
            plt.close()
        
        if pm_direction_cols:
            # PM Peak Volume Heatmap
            plt.figure(figsize=(12, 8))
            pivot_pm = gdf_link.pivot_table(
                values=pm_direction_cols,
                index='STREET',
                aggfunc='mean'
            ).head(20)  # Top 20 streets for visibility
            
            sns.heatmap(pivot_pm, cmap='YlOrRd', annot=True, fmt='.0f')
            plt.title('PM Peak Volume by Direction (Top 20 Streets)')
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "pm_peak_volume_heatmap.png"))
            plt.close()
        
        # Create alternative heatmaps using total volumes if directional data is not available
        if not am_direction_cols or not pm_direction_cols:
            # Total peak volumes heatmap
            plt.figure(figsize=(12, 8))
            pivot_total = gdf_link.pivot_table(
                values=['PkAM_1314', 'PkPM_1314'],
                index='STREET',
                aggfunc='mean'
            ).head(20)  # Top 20 streets for visibility
            
            sns.heatmap(pivot_total, cmap='YlOrRd', annot=True, fmt='.0f')
            plt.title('Peak Volume by Time Period (Top 20 Streets)')
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "peak_volume_heatmap.png"))
            plt.close()
    
    # 5. Create interactive maps for web visualization
    interactive_maps = create_interactive_map(gdf_link)
    
    # 6. Save metrics as JSON for the web app
    metrics_path = os.path.join(REPORTS_TRAFFIC_FLOW_DIR, "summary_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved summary metrics to {metrics_path}")
    
    print("Traffic flow analysis complete!")
    
    return gdf_link

def main():
    """Main execution function"""
    print("Starting Traffic Flow Analysis...")
    
    # Call the analyze_traffic_flow function
    gdf_link = analyze_traffic_flow()
    
    print("\nTraffic Flow Analysis completed successfully!")
    print("\nRun the Streamlit app to view the analysis.")

if __name__ == "__main__":
    main() 