# Ensemble DNA methylation survival clock

[![Python 3.9](https://img.shields.io/badge/python-3.9-3776AB.svg)](requirements.txt)
[![Release](https://img.shields.io/github/v/release/dante-yg/Ensemble-Survival-Clock?color=2F5D8A)](https://github.com/dante-yg/Ensemble-Survival-Clock/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-2F5D8A.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.mad.2026.112240-2F5D8A.svg)](https://doi.org/10.1016/j.mad.2026.112240)

This repository contains the code and trained model for an ensemble DNA methylation clock that predicts all-cause mortality. Five survival models built on 190 CpGs, selected by elastic-net Cox regression, are combined by a neural network meta-learner. The model was trained in the Framingham Heart Study and validated in the Women's Health Initiative.

<p align="center">
  <img src="docs/pipeline.svg" alt="Pipeline: DNA methylation, elastic-net Cox feature selection, five base survival models, neural network meta-learner, ensemble risk score" width="100%">
</p>

## Performance

| Cohort | Evaluation role | *n* | C-index (95% CI) | Time-dependent AUROC | Integrated Brier score |
|---|---|---:|---|---|---|
| Framingham Heart Study | Held-out test set (20%) | 700 | 0.838 (0.809–0.861) | 0.851 (5 y), 0.850 (10 y), 0.804 (12 y) | 0.065 |
| Women's Health Initiative | External validation | 1,862 | 0.703 (0.687–0.720) | 0.646 (10 y), 0.742 (15 y), 0.767 (20 y) | 0.147 |

## Usage

1. **01_feature_selection.ipynb** splits the FHS data into training (60%), validation (20%) and test (20%) sets, standardises the CpGs on the training set, and selects CpGs with cross-validated elastic-net Cox regression. The list used in the paper is included as `coxnet_selected_cpgs_190.csv`.
2. **02_train_ensemble.ipynb** trains the five base survival models (random survival forest, gradient boosting, kernel survival SVM, Weibull AFT and MTLR) and the neural network meta-learner, then rescales the ensemble risk score to years of age. After the first cell has run, set `RUN_ID` in steps 3–5 of the notebook to the run folder it created.
3. **03_run_inference.ipynb** (or **score_samples.py** from the command line) loads the trained ensemble from `pretrained_models/` and scores new samples. The model files are described in [pretrained_models/README.md](pretrained_models/README.md).

## Scoring your own samples

From the command line, pass a `.csv`, `.tsv` or `.parquet` file with one row per sample and one column per CpG (beta values):

```bash
python score_samples.py --input my_methylation.csv --output predictions.csv
```

Or interactively in `03_run_inference.ipynb`:

```python
import pandas as pd

df = pd.read_csv("my_methylation.csv")
predictions = pipeline.predict(df)
```

The output contains the ensemble risk score (`ensemble_risk`), the score rescaled to years (`calibrated_age_pred`) and the prediction of each base model. If the input has an `age` column, `age_acceleration` is added. CpGs missing from the input are set to 0.5.

The rescaled score is a mortality risk score expressed on an age scale, not an estimate of biological age.

## Requirements

- Python 3.9 with TensorFlow 2.10
- scikit-learn, scikit-survival, lifelines, PyTorch, pycox, torchtuples, numpy, pandas, pyarrow, h5py and joblib
- Jupyter

Exact versions are pinned in [requirements.txt](requirements.txt).

## Data

The FHS (phs000007.v35.p1) and WHI (phs000200.v12.p3) data are controlled-access and are not included. Access can be requested through [dbGaP](https://www.ncbi.nlm.nih.gov/gap/).

The training notebooks read parquet files from a `data/` folder: one row per participant, with an ID, age, sex, follow-up time in years, a death indicator (`event`) and one column per CpG (beta values). Scoring new samples with `03_run_inference.ipynb` needs only the CpG columns.

## Getting started

1. Clone the repository:
   ```bash
   git clone https://github.com/dante-yg/Ensemble-Survival-Clock.git
   cd Ensemble-Survival-Clock
   ```
2. Create an environment and install the packages:
   ```bash
   conda create -n ensemble-clock python=3.9
   conda activate ensemble-clock
   pip install -r requirements.txt
   ```
3. Start Jupyter from the repository folder and open `03_run_inference.ipynb`:
   ```bash
   jupyter notebook
   ```

## Citation

If you use this code or model, please cite:

Muthukumar YG, Vijayakumar KA, Cho GW. Ensemble DNA methylation clock demonstrates immune-metabolic aging signatures associated with mortality. *Mechanisms of Ageing and Development*. 2026;233:112240. https://doi.org/10.1016/j.mad.2026.112240

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
