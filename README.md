# INFO411 Data Mining Project – Online Retail Analysis

## Overview

This project explores customer behaviour using the UCI Online Retail dataset. The analysis includes:

* Data preprocessing and feature engineering
* Customer segmentation using clustering
* Churn prediction using neural neworks
* Time series forecasting

The goal is to extract meaningful insights and build predictive models to support business decision-making.

---

## Project Structure

This repository contains four main files:

### 1. Data Preprocessing

**File:** `projectDataProcessing.Rmd`

* Cleans raw transaction data
* Creates customer-level features (Recency, Frequency, Monetary, etc.)
* Generates the final dataset used across all models

---

### 2. Clustering (Customer Segmentation)

**File:** `clustering.Rmd`

* Applies K-means clustering
* Determines optimal number of clusters (Elbow + Silhouette)
* Profiles customer segments
* Evaluates clusters

---

### 3. Classification (Churn Prediction)

**File:** `neural.ipynb`

* Builds machine learning models to predict churn
* Includes Neural Network and comparison models
* Evaluates performance using metrics such as accuracy, precision, recall, and F1-score

---

### 4. Time Series Forecasting

**File:** `timeSeries.ipynb`

* Analyses monthly revenue trends
* Applies smoothing techniques (Moving Average, SES)
* Builds ARIMA model for forecasting

---

## Requirements

### R Studio (for .Rmd files)

### Google Colab (for .ipynb files)


## Authors

* 8233718
* 8887901
* 9091488

---
