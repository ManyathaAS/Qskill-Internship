# QSkill Internship — Slab 2 (Intermediate) · Data Science Projects

Three end-to-end machine-learning projects, each in its own Jupyter notebook.

| # | Project | Notebook | Main techniques |
|---|---------|----------|-----------------|
| 1 | E-Commerce Customer Churn Prediction | `notebooks/Task1_Churn_Prediction.ipynb` | cleaning, EDA, feature engineering & selection, class imbalance, model comparison, tuning, precision / recall / F1 / ROC-AUC, confusion matrices, churn drivers, recommendations |
| 2 | Time Series Sales Forecasting | `notebooks/Task2_Sales_Forecasting.ipynb` | date handling, trend / seasonality / anomaly analysis, lag + rolling + calendar features, chronological split, baselines vs ML, recursive 90-day forecast, MAE / RMSE / MAPE |
| 3 | End-to-End Fraud Detection & Risk Analysis | `notebooks/Task3_Fraud_Detection.ipynb` | EDA of transaction patterns, imbalance analysis, model comparison & tuning, PR-AUC, threshold & cost analysis, false-positive / false-negative analysis, feature importance, risk tiers, recommendations |

## Project structure
```
├── data/
│   ├── ecommerce_churn.csv        # Task 1 data
│   ├── store_sales.csv            # Task 2 data
│   └── paysim_transactions.csv    # Task 3 data
├── notebooks/                     # the three notebooks (already executed, outputs included)
├── outputs/                       # files produced by the notebooks (forecast, flagged transactions)
├── src/generate_data.py           # script that created the practice datasets
├── requirements.txt
└── README.md
```

## How to run
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook                  # open the notebooks folder and run each notebook top to bottom
```
Notebooks use relative paths (`../data/...`), so open them from inside the `notebooks/` folder (Jupyter does this automatically).

## About the datasets
The bundled CSVs are **practice datasets that mirror the column layout of well-known public Kaggle datasets**:

| Task | Public dataset with the same columns | Expected file name in `data/` |
|------|--------------------------------------|-------------------------------|
| 1 | *E-Commerce Customer Churn* (`E Commerce Dataset.xlsx`, sheet `E Comm`) | `ecommerce_churn.csv` |
| 2 | *Store Item Demand Forecasting Challenge* (`train.csv`) | `store_sales.csv` |
| 3 | *PaySim Synthetic Financial Dataset* | `paysim_transactions.csv` |

To run the projects on the real data, download the Kaggle file, save it in `data/` under the name above (for Task 1, save the sheet as CSV), and re-run the notebook. Column names are identical, so no code changes are needed. Task 3 automatically samples very large files for speed.

Regenerate the practice data any time with `python src/generate_data.py`.

## Reproducibility
All random seeds are fixed, so re-running gives the same numbers. Modelling uses scikit-learn only (no extra ML libraries required).
