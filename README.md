# Predicting Delivery Times

Exploratory analysis and CatBoost modelling of food delivery duration, implemented as two marimo notebooks.

## Inspect the saved runs

Completed notebook runs, including code, commentary, tables and plots, are available as HTML files:

- `__marimo__/explore.html` — data exploration, cleaning and feature engineering
- `__marimo__/train.html` — validation, feature ablations, error analysis, SHAP interpretation and final predictions

## Setup

Use `uv` to install dependencies:

```
uv sync
```

Before running `explore.py`, copy the following files from the [original assignment repository](https://github.com/1001-ai/ml-take-home-assignment/tree/main/dataset) into `dataset/`, keeping their filenames unchanged:

- `train.csv`
- `test.csv`
- `Sample_Submission.csv`

The sample submission is used to validate the generated CSV's column names and ID ordering. The `processed/` and `figures/` directories are supplied empty and are populated by the notebooks.

## Run the notebooks

Run **`explore.py` first**, followed by **`train.py`**.

### Interactive execution

Open the exploration notebook:

```bash
uv run marimo edit explore.py
```

Execute all cells and wait for completion. This writes the cleaned and feature-engineered training and test data to parquet files in `processed/`.

Stop the notebook server with `Ctrl+C`, then open the training notebook:

```bash
uv run marimo edit train.py
```

Execute all cells. This runs the median baseline and CatBoost cross-validation experiments, produces the diagnostic plots, retrains the selected model on the full cleaned training set, and generates test predictions.