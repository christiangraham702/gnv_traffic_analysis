# Traffic Analysis Project

This project analyzes traffic data to predict intersection delays and visualize traffic patterns in an interactive web application.

## Project Structure

```
project/
│
├── data/
│   ├── raw/          # Original CSV files
│   └── clean/        # Cleaned and processed data
│
├── models/           # Serialized models and scalers
│
├── notebooks/        # Jupyter notebooks for analysis steps
│
├── app/              # Streamlit web application
│   └── assets/       # Images, CSS, etc.
│
├── reports/          # Analysis outputs
│   ├── figures/      # Generated plots and visualizations
│   └── tables/       # Data tables and summaries
│
└── env/              # Environment configuration
```

## Setup

### Using Conda

```bash
conda env create -f env/environment.yml
conda activate traffic-analysis
```

### Using Pip

```bash
pip install -r env/requirements.txt
```

## Running the Application

Once dependencies are installed:

```bash
streamlit run app/streamlit_app.py
```

## Data Sources

The analysis uses traffic data from Florida Department of Transportation (FDOT):
- `traffic_data.csv`: Intersection-level traffic data
- `lots_traffic_data.csv`: Link/segment traffic data

## Analysis Pipeline

1. Data ingestion and cleaning
2. Feature engineering
3. Exploratory data analysis
4. Predictive modeling
5. Interactive visualization 