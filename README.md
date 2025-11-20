# Solar Radiation Climate Change Impact Analysis

## 📋 Project Overview

This project analyzes the impact of climate change related to solar radiation by collecting data from two primary sources: NASA POWER API and Open-Meteo API. The goal is to analyze and compare multi-dimensional climate data to understand patterns and trends in climate change.

## 🗂️ Project Structure

```
Climate Change/
│
├── README.md                           # Project documentation (this file)
│
├── Dataset/                            # Data folder
│   ├── nasa_daily_parameters.csv       # NASA POWER API parameter list
│   ├── nasa_power_data.csv             # Data from NASA POWER API
│   ├── not_clean_climate_data.csv      # Raw uncleaned climate data
│   ├── openmeteo_climate_data.csv      # Data from Open-Meteo API
│   └── openmeteo_daily_parameters.csv  # Open-Meteo API parameter list
│
└── Jupyter Notebook/                   # Notebook folder
    └── explore_api.ipynb               # Notebook for API exploration and data collection
```

## 📍 Study Area

- **Location**: Bangkok, Thailand
- **Coordinates**: 
  - Latitude: 13.736717°N
  - Longitude: 100.523186°E

## 🚀 Getting Started

1. Open `Jupyter Notebook/explore_api.ipynb` to see API exploration and data collection
2. Check data files in the `Dataset/` folder for collected data
3. Use parameter files (.csv) as reference guides for selecting data to analyze
