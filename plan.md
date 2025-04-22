Below is a step‑by‑step “playbook” your LLM‑powered coding agent can follow from raw CSVs to a polished, share‑worthy web page full of interactive maps and charts.
It’s written as a checklist with explicit inputs, outputs, and guardrails so the agent can execute autonomously yet reproducibly.

⸻

0 Project Skeleton & Conventions

Folder	Purpose
data/raw/	lots_traffic_data.csv, traffic_data.csv, original PDFs (keep read‑only)
data/clean/	cleaned & merged parquet/CSV files
models/	serialized models (.pkl or .joblib) + scalers
notebooks/	one exploratory notebook per major step (EDA, feature eng, modeling)
app/	final web app (streamlit_app.py, HTML, JS, CSS, assets/)
reports/	figures/, tables/, Markdown summary
env/	environment.yml or requirements.txt

Libraries (Py ≥3.10): pandas, numpy, pyarrow, geopandas, folium, plotly, scikit‑learn, shap, streamlit, leaflet.js (bundled via Folium).

Set global seed: np.random.seed(42).

⸻

1 Ingest & Schema Audit
	1.	Load files

df_int = pd.read_csv("data/raw/traffic_data.csv")        # intersection‑level
df_link = pd.read_csv("data/raw/lots_traffic_data.csv")  # larger link/segment set


	2.	Print .info() / .describe(); export to reports/tables/raw_schema.md.
	3.	Assert uniqueness keys
	•	df_int: ["INTERSECT","Period","Time"]
	•	df_link: if not unique, aggregate by Station_ID + Year.

⸻

2 Data Cleaning & Harmonisation
	1.	Type coercion
	•	Strip % then astype(float) on heavy‑vehicle columns.
	•	pd.to_numeric(..., errors="coerce") for greens / lane counts.
	2.	Datetime index (optional): convert Year, Period, Time into a single timestamp.
	3.	Missing values
	•	<5 % → impute with median.
	•	≥5 % or critical → drop row & log to reports/tables/dropped_rows.csv.
	4.	Save cleaned copies to data/clean/df_int.parquet & df_link.parquet.

⸻

3 Feature Engineering

3.1 Intersection Dataset (df_int)

New Feature	Formula
Total_Vol	sum of all turning‑movement counts
EW_Vol	(EB_L + EB_Th + EB_R) + (WB_L + WB_Th + WB_R)
NS_Vol	(NB_L + NB_Th + NB_R) + (SB_L + SB_Th + SB_R)
EW_to_NS	EW_Vol / NS_Vol
Green_EW	Max_green_EBL + Max_green_EBT (or mean)
Green_to_Demand_EW	Green_EW / EW_Vol
Ped_Load	Ped_clearance_EW + Ped_clearance_NS

Create target columns:

df_int["High_Delay"] = (df_int["Delay_s_veh"] > 60).astype(int)

3.2 Link Dataset (df_link)
	1.	Restrict to a reference year (e.g., 2013‑2014) or add Year as categorical.
	2.	Aggregate directional peaks:

df_link["Peak_Ratio"] = df_link["PkPM_1314"] / df_link["PkAM_1314"]


	3.	Encode heavy‑vehicle share, ADT, AMD1/PMD1 for ML.

⸻

4 Exploratory Data Analysis (EDA)
	1.	Univariate histograms (Plotly) for counts, greens, delay.
	2.	Bivariate
	•	Scatter Delay_s_veh vs Total_Vol, colored by intersection.
	•	Boxplots of delay by Period.
	3.	Correlation heatmap with Pearson / Spearman; save to reports/figures/corr_heatmap.png.
	4.	Document key findings in reports/eda_summary.md.

⸻

5 Modeling Pipeline

Goal A Intersection‑level classification (High vs Low delay)
Goal B (Optional) Regression on Delay_s_veh.

5.1 Train / Validation Split
	•	Stratify by High_Delay and INTERSECT (train_test_split(..., test_size=0.2, stratify=df_int["INTERSECT"])).

5.2 Baseline Model

from sklearn.tree import DecisionTreeClassifier
model0 = DecisionTreeClassifier(random_state=42, max_depth=None)

5.3 Improved Model: Random Forest

from sklearn.ensemble import RandomForestClassifier
param_grid = {
    "n_estimators":[200,400,800],
    "max_depth":[None,6,12],
    "min_samples_leaf":[1,3,5]
}
GridSearchCV(..., cv=5, scoring="f1")

5.4 Metrics
	•	Accuracy, Precision, Recall, F1, ROC‑AUC; confusion matrix heatmap.

5.5 Feature Importance & SHAP
	•	Save bar chart plus SHAP summary to reports/figures/feature_importance.png.

5.6 Persistence

import joblib
joblib.dump(best_model, "models/high_delay_rf.pkl")



⸻

6 Scenario Generator & Transfer Learning

6.1 Leverage df_link for Synthetic Scenarios
	1.	Cluster link segments (KMeans(n_clusters=5)) on ADT, Peak_Ratio, Heavy_%.
	2.	Identify cluster centroids that resemble each intersection’s Total_Vol & Peak_Ratio.
	3.	Generate N synthetic rows per intersection by sampling from matching cluster distributions.

6.2 Predict Delay Class
	•	Feed synthetic rows into best_model.predict_proba.
	•	Record probability of high delay; flag scenarios that drop below 0.3.

⸻

7 Interactive Web App / Dashboard

7.1 Tech Choice

Use Streamlit for speed + Folium/Leaflet for maps.

7.2 Page Layout

Section	Elements
Header	Title, brief description
Model Performance	Metrics table, confusion matrix image
Feature Importance	SHAP bar chart (Plotly)
Interactive Map	Folium map centered on Gainesville:
→ Circle markers at intersection coords (from df_int), color‑coded by predicted high/low delay.	
→ Popup with Delay_s_veh, total volume, heavy %	
Scenario Sandbox	Slider inputs (traffic volume, green extension). On submit → model prediction + updated map.
Download	Button to export cleaned dataset and HTML report.

7.3 Implementation Hints

import streamlit as st
import folium
from streamlit_folium import st_folium

Embed Folium map in Streamlit; use Plotly for dynamic charts.

7.4 Build & Launch

streamlit run app/streamlit_app.py



⸻

8 Documentation & Reproducibility
	1.	README.md with setup, run, and dataset description.
	2.	Environment file (environment.yml):

name: traffic-analysis
channels: [conda-forge]
dependencies:
  - python=3.10
  - pandas
  - geopandas
  - scikit-learn
  - shap
  - folium
  - streamlit
  - plotly


	3.	Data provenance: note raw → clean hashes.
	4.	Model cards: brief YAML outlining data, metrics, caveats (models/model_card.yml).

⸻

9 Quality & Ethical Checklist
	•	✔️ Validate geocoding accuracy (no swapped lat/long).
	•	✔️ Confirm no personally identifiable information.
	•	✔️ Communicate model limitations (small intersection sample, synthetic assumptions).
	•	✔️ Provide links to original FDOT sources.

⸻

🔚 Deliverables
	1.	streamlit_app.py — interactive dashboard
	2.	models/high_delay_rf.pkl — trained classifier
	3.	data/clean/*.parquet — harmonised datasets
	4.	reports/ — figures, tables, EDA & methodology markdown
	5.	README.md + environment file

Follow this road‑map and your LLM agent will take raw CSVs ➜ cleaned data ➜ predictive models ➜ a slick, map‑rich webpage—the full analytics pipeline in one automated run.