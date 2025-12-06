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
* **Algo_Trading_ALL.ipynb**: ML Python code outsourced to DataBricks.
* **ProTrader7.py**: Streamlit script.
* **Requirements.txt**: List of libraries needed to run the app.
