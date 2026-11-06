# Pretrained models

The trained ensemble used in the paper. All files are loaded by `03_run_inference.ipynb`.

| File | What it is |
|---|---|
| `cpg_scaler.joblib` | Standardises the 190 CpG beta values before scoring |
| `base_models/rsf_full.joblib` | Random survival forest |
| `base_models/gbm_full.joblib` | Gradient boosting survival model |
| `base_models/svm_full.joblib` | Kernel survival SVM |
| `base_models/weibull_full.joblib` | Weibull AFT model |
| `base_models/mtlr_full.pt` | MTLR model (PyTorch) |
| `pred_scalers/pred_scaler_*.joblib` | Standardise each base model's risk score before it goes to the meta-learner |
| `stacking_meta_learner.keras` | Neural network meta-learner (Keras 2.10) |
| `age_regressor.joblib` | Linear rescaling of the ensemble risk score to years |

The models need the package versions in [requirements.txt](../requirements.txt), in particular TensorFlow 2.10 and scikit-learn 1.3.2.
