# Monthly Sales Forecasting with Machine Learning

## Overview

This academic group project explored monthly sales forecasting for a business case involving Siemens Smart Infrastructure in Germany. The objective was to investigate how historical sales patterns and time-series machine learning could support more consistent forecasting and business planning.

The project was completed for **Business Cases with Data Science** in the **MSc in Data Science and Advanced Analytics** at **NOVA Information Management School (NOVA IMS)** in **March 2025**.

## Business Problem

Manual sales forecasts can require substantial effort and may be difficult to standardize across product groups. This case examined whether a data-driven forecasting workflow could help inform inventory planning, resource allocation, and operational decisions.

## Data and Scope

The original academic case used Siemens-provided business sales records and external economic and market indicators. These sources were used for exploratory analysis and to investigate potential forecasting features.

**Data confidentiality:** The underlying company datasets, unredacted outputs, exact sales figures, product-level forecasts, and original report are not distributed in this public-facing project description. Reproduction of results from the original case requires authorized access to the source data.

## Methodology

The analysis followed the CRISP-DM framework and included:

1. **Data understanding and exploration:** Review historical sales patterns, market indicators, trends, seasonality, and variability across product groups.
2. **Data preparation:** Aggregate sales to monthly frequency, investigate missing values and outliers, and prepare the time series for modeling.
3. **Feature engineering:** Develop lagged sales features, rolling statistics, and calendar features such as month, quarter, and year.
4. **Forecasting models:** Develop XGBoost regression models for product groups with different sales characteristics. The accompanying project report also discusses Prophet; the supplied implementation notebook primarily demonstrates XGBoost.
5. **Model evaluation:** Use time-series cross-validation and error measures such as root mean squared error (RMSE) to examine forecasting performance.
6. **Forecast generation:** Produce experimental forecasts over a ten-month horizon.

## Tools and Technologies

Python, Jupyter Notebook, Pandas, NumPy, XGBoost, Scikit-learn, Statsmodels, Matplotlib, Seaborn, and Bayesian Optimization.

## Findings and Limitations

The study selected XGBoost for the final experimental forecasts and highlighted the importance of adapting feature engineering and models to differences in product-level sales patterns.

The work was an **academic prototype**, not a production deployment or a validated improvement over Siemens' existing forecasting process. Its performance and business impact should not be interpreted as established without additional out-of-sample testing and independent validation. Some methodological and implementation details would require revision before reusing the original notebook for live forecasts.

## Repository contents

- `monthly_sales_forecasting.ipynb`: public reference notebook with outputs, company-specific narrative, original product-group assignments, and original dataset paths removed
- `forecast_helpers.py`: helper functions for plotting, statistical diagnostics, and XGBoost experiments
- `requirements.txt`: Python dependencies (versions not pinned; historical compatibility not guaranteed)
- `.gitignore`: prevents common data files, credentials, and local outputs from being committed

The reference notebook **does not run without authorized datasets and schema adaptation**. Original performance is not independently verified; output redaction does not establish permission to publish source code.

## Reproducibility and Publication

The company-provided datasets are not public. Any code or demonstrations added to this repository should exclude confidential or commercially sensitive data and be published only with appropriate authorization. A separately labeled synthetic-data demonstration may be used to illustrate the general methodology without reproducing company figures.

## Project Team

NOVA IMS, Group E (March 2025):

- Jude Gbenimako
- Tomás Ribeiro
- Elcano Gaspar
- Afonso Silva
- Rui Reis

## Disclaimer

This is a student academic project description. It does not represent an official Siemens publication, endorsement, or operational product.
