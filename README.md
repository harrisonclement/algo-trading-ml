# Algorithmic Trading & Machine Learning Framework

## Overview
This project implements a comprehensive machine learning pipeline for algorithmic trading on the S&P 500 (SPY). It features an end-to-end workflow including data ingestion via `yfinance`, feature engineering (momentum, volatility), a "Grand Sweep" model evaluation (Linear Regression, SVR, XGBoost, Random Forest, etc.), and a realistic backtesting engine.

Experiment tracking is handled via **MLflow** (integrated with Databricks or local fallback).

## Features
* **Automated Data Pipeline**: Fetches daily OHLCV data and handles time-series splitting.
* **Feature Engineering**: Generates rolling averages, volatility, and momentum indicators.
* **Model Sweep**: Trains and evaluates 7+ regression models with hyperparameter tuning.
* **Backtesting Engine**: Simulates trading strategies and compares cumulative returns against a Buy & Hold baseline.
* **MLflow Integration**: Logs metrics (RMSE, MAE), parameters, and artifacts (equity curves).

## Project Structure
* `src/`: Core logic for data, features, training, and backtesting.
* `notebooks/`: Exploratory analysis and prototyping.
* `main.py`: CLI entry point for executing pipelines.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/algo-trading-ml.git](https://github.com/yourusername/algo-trading-ml.git)
    cd algo-trading-ml
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Credentials (Optional for Databricks):**
    Set environment variables if using Databricks MLflow. Otherwise, it defaults to local logging.
    ```bash
    export DATABRICKS_HOST="your-host"
    export DATABRICKS_TOKEN="your-token"
    ```

## Usage

**Run the Model Sweep:**
Train all defined models and log results to MLflow.
```bash
python main.py --mode train --user your_email@example.com
