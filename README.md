# DNA Methylation Survival Analysis Pipeline

This repository implements a stacked ensemble survival-analysis pipeline for predicting mortality and age-related outcomes from DNA methylation data. The pipeline selects key methylation CpG features using an elastic-net Cox model, trains five base survival learners (Random Survival Forest, Gradient Boosting Machine, Fast Kernel SVM, Weibull AFT, and MTLR), trains a DeepSurv-style CoxMLP stacking meta-learner, and evaluates performance on internal splits and an external validation cohort.

## Data
The input DNA methylation parquet files are not included in this repository and must be obtained directly from the original cohorts (FHS and WHI).

## Setup
- **Python Version**: Python 3.9 is targeted.
- **Dependencies**: Install the required packages via:
  ```bash
  pip install -r requirements.txt
  ```

## Directory Layout
The notebooks automatically create and structure files within the following layout:
- `data/`: Contains the raw methylation parquet datasets (must be populated by the user).
- `FHS60_pipeline/`: Stores the stacking meta-learner assets.
  - `splits/`
  - `scalers/`
  - `predictions/`
  - `savedmodels/`
  - `logs/`
- `Ensemble_model/`: Stores the base learner outputs and results.
  - `saved_models/`
  - `logs/`
  - `outputs/csv/`
  - `outputs/figures/`

## Run Order
1. Run [feature_selection.ipynb](feature_selection.ipynb) top to bottom. This splits the FHS dataset, scales features, fits the elastic-net Cox model, and saves the list of selected CpG probes to a non-zero coefficients CSV.
2. Copy and rename that generated coefficients CSV to `./coxnet_selected_cpgs_190.csv` (which is the filename expected by the next step).
3. Run [ensemble_stacking_pipeline.ipynb](ensemble_stacking_pipeline.ipynb) top to bottom. After executing the first pipeline cell, identify the generated run-id folder name (e.g., `FHS60_YYYYMMDDHHMMSS`) printed in the output, and set the hardcoded `RUN_ID` in the subsequent cells to match this run-id.
