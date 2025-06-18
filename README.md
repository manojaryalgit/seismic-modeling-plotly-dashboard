# Seismic Modeling Plotly Dashboard
## on Collaboration with: Bishnu Prasad Sharma (https://github.com/bishnu820)

An interactive web application for analyzing earthquake data in Nepal using machine learning models for prediction and risk assessment.

![Seismic Dashboard](https://github.com/manojaryalgit/seismic-modeling-plotly-dashboard/raw/main/assets/dashboard_preview.png)

## Features

- **Interactive Maps**: Explore earthquake locations with heatmaps and marker overlays
- **Temporal Analysis**: Analyze earthquake patterns over time
- **Feature Analysis**: Understand key earthquake indicators and parameters
- **Machine Learning Models**: Predictive models for earthquake magnitude analysis
- **Statistical Analysis**: Advanced statistics on earthquake data
- **Risk Assessment**: Tools to evaluate seismic risk factors

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository
   ```
   git clone https://github.com/manojaryalgit/seismic-modeling-plotly-dashboard.git
   cd seismic-modeling-plotly-dashboard
   ```
2. Create a virtual environment (recommended)
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Navigate to the project directory
2. Run the application
   ```
   python app.py
   ```
3. Open your browser and go to http://127.0.0.1:8051/

## Project Structure

- `app.py`: Main application entry point
- `requirements.txt`: Project dependencies
- `assets/`: CSS styling and other static files
- `data/`: Contains earthquake datasets
- `models/`: Trained machine learning models
- `pages/`: Dashboard pages for different analyses

## Note on Models

The trained models were created using data available up to June 2025 and include:
- Ridge Regression
- Random Forest
- XGBoost
- CatBoost

## License

This project is licensed for educational and research purposes only.
