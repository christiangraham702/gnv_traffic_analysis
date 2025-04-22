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

## Deployment Options

### Streamlit Cloud (Recommended)

The easiest way to deploy this dashboard:

1. Ensure your code is in a GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository
5. Set the path to `app/streamlit_app.py`
6. Click "Deploy"

### Heroku Deployment

This repository includes configuration for Heroku deployment:

```bash
# Login to Heroku
heroku login

# Create a new Heroku app
heroku create your-app-name

# Push to Heroku
git push heroku main

# Open the app
heroku open
```

### Local Development

For local development or temporary sharing:

```bash
# Start the app locally with the helper script
python run_streamlit.py

# Or manually
streamlit run app/streamlit_app.py
```

## Streamlined Dependencies

The `requirements.txt` file has been optimized to include only the necessary packages for deployment. 