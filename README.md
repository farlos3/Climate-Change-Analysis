# Climate Change Analysis Project

![Data Pipeline](data%20pipeline.png)

## Overview
This project analyzes climate data and forecasts temperature and other environmental parameters using machine learning models. It includes data engineering, feature selection, model training, and deployment via APIs and orchestration tools.

## Project Structure
- **Dataset/**: Raw and processed climate data files (CSV).
- **Notebook/**: Jupyter notebooks for EDA, feature engineering, model comparison, and API exploration.
- **Model/**: Model training notebooks and saved model weights.
- **backend-project/**: Backend services, including FastAPI, Airflow, and Docker Compose setup.
- **src/**: Python scripts for data preparation, ETL, and feature engineering.

## Key Components
- **Data Preparation**: Cleaning and transforming climate data for modeling.
- **Feature Engineering**: Selecting and creating features relevant for forecasting.
- **Model Training**: Training models (LightGBM, Random Forest, XGBoost, LSTM) for temperature prediction.
- **Model Comparison**: Comparing model performance and visualizing results in notebooks.
- **API Deployment**: Serving predictions via FastAPI endpoints.
- **Orchestration**: Using Airflow for pipeline automation and Docker Compose for service management.
- **Reverse Proxy**: nginx configuration for routing API and Airflow services.

## How to Run
1. **Clone the repository**
2. **Install dependencies** (see `requirements.txt` in relevant folders)
3. **Start services**:
   - Use Docker Compose: `docker-compose up --build`
   - Access FastAPI docs at `http://localhost:8000/docs`
   - Access Airflow UI at `http://localhost:8080`
4. **Run notebooks** for EDA, feature selection, and model comparison.

## Airflow Admin Access
To access the Airflow web UI, you need an admin account. The default admin user is created automatically when starting Airflow with Docker Compose.

To find the generated admin password, run:

```bash
docker-compose logs airflow_standalone | grep -A2 -B2 "Password for user" | tail -4
```

This will display the username and password for the Airflow admin account in the logs. Use these credentials to log in at `http://localhost:8080`.

## Notebooks
- `Notebook/feature_selection.ipynb`: Feature selection for modeling
- `Notebook/feature_t2m.ipynb`: T2M feature engineering
- `Model/Ensemble&Compare.ipynb`: Model comparison and visualization
- `Notebook/analysis.ipynb`: Data analysis and exploration

## API Endpoints
- FastAPI endpoints for prediction and feature ingestion (see `backend-project/app/main.py`)

## Orchestration & Automation
- Airflow DAGs for pipeline automation (`backend-project/airflow/dags/`)
- Docker Compose for multi-service orchestration (`backend-project/docker-compose.yml`)
