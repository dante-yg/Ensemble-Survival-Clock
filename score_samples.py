"""Command-line inference script for the Ensemble DNA methylation survival clock.

Example:
    python score_samples.py --input my_methylation.csv --output predictions.csv
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import h5py
import joblib
import numpy as np
import pandas as pd
import sklearn.base
import sksurv.ensemble.survival_loss
import tensorflow as tf
import torch
from pycox.models import MTLR as PycoxMTLR
from torchtuples import optim as ttoptim
from torchtuples.practical import MLPVanilla


class DummySurvivalEstimator(sklearn.base.BaseEstimator):
    def predict(self, X):
        val = getattr(self, "value_", getattr(self, "value", 0.0))
        if isinstance(val, (np.ndarray, list)):
            return np.full(X.shape[0], val[0])
        return np.full(X.shape[0], val)


sksurv.ensemble.survival_loss.DummySurvivalEstimator = DummySurvivalEstimator


class WeibullAFTAdapter:
    def __init__(self, penalizer=0.1, l1_ratio=0.0):
        self.penalizer = float(penalizer)
        self.l1_ratio = float(l1_ratio)
        self.model = None

    def get_params(self, deep=True):
        return {"penalizer": self.penalizer, "l1_ratio": self.l1_ratio}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def predict(self, X):
        X = np.asarray(X)
        cols = [f"x{i}" for i in range(X.shape[1])]
        exptime = self.model.predict_expectation(pd.DataFrame(X, columns=cols))
        return -np.asarray(exptime).ravel()


setattr(sys.modules["__main__"], "WeibullAFTAdapter", WeibullAFTAdapter)


def pred_risk(model, X):
    X = np.asarray(X)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X)).ravel()
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel()
    raise RuntimeError(f"No prediction method for {type(model)}")


def mtlr_risk_scores(model, X_np, cuts):
    surv_df = model.predict_surv_df(X_np)
    if hasattr(np, "trapezoid"):
        return (-np.trapezoid(surv_df.values, x=cuts, axis=0)).astype(np.float64)
    return (-np.trapz(surv_df.values, x=cuts, axis=0)).astype(np.float64)


def patch_sklearn_estimator(model):
    if model is None:
        return
    if not hasattr(model, "monotonic_cst"):
        try:
            model.monotonic_cst = None
        except Exception:
            pass
    for attr in ("_loss", "loss_"):
        loss_obj = getattr(model, attr, None)
        if loss_obj is not None and not hasattr(loss_obj, "link"):
            from sklearn._loss.link import IdentityLink
            try:
                loss_obj.link = IdentityLink()
            except Exception:
                pass
    if hasattr(model, "estimators_"):
        ests = model.estimators_.ravel() if isinstance(model.estimators_, np.ndarray) else model.estimators_
        for est in ests:
            patch_sklearn_estimator(est)
    if hasattr(model, "init_"):
        patch_sklearn_estimator(model.init_)


def load_keras_model_robust(model_path):
    model_path = Path(model_path)
    try:
        return tf.keras.models.load_model(model_path.as_posix(), compile=False)
    except Exception:
        pass

    h5_path = model_path
    temp_h5_created = False
    if model_path.suffix == ".keras":
        h5_path = model_path.with_suffix(".h5")
        if not h5_path.exists():
            shutil.copy(model_path, h5_path)
            temp_h5_created = True
    try:
        with h5py.File(h5_path, "r") as f:
            weight_keys = list(f["model_weights"].keys())
        dense_keys = sorted(
            [k for k in weight_keys if "dense" in k],
            key=lambda x: int(x.split("_")[1]),
        )
        inputs = tf.keras.Input(shape=(5,), name="input_52")
        x = tf.keras.layers.Dense(128, activation="relu", name=dense_keys[0])(inputs)
        x = tf.keras.layers.Dropout(0.35)(x)
        x = tf.keras.layers.Dense(64, activation="relu", name=dense_keys[1])(x)
        x = tf.keras.layers.Dropout(0.35)(x)
        x = tf.keras.layers.Dense(128, activation="relu", name=dense_keys[2])(x)
        x = tf.keras.layers.Dropout(0.35)(x)
        outputs = tf.keras.layers.Dense(1, activation="linear", name=dense_keys[3])(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.load_weights(h5_path.as_posix())
        return model
    finally:
        if temp_h5_created and h5_path.exists():
            try:
                os.remove(h5_path)
            except Exception:
                pass


def load_input_table(input_path, cpg_list, id_cols=("Share_ID", "Patient_ID", "Subject_ID", "age")):
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(
            input_path,
            thrift_string_size_limit=2_000_000_000,
            thrift_container_size_limit=2_000_000_000,
        )
        cols = [c for c in cpg_list if c in pf.schema.names]
        for col in id_cols:
            if col in pf.schema.names and col not in cols:
                cols.append(col)
        return pf.read(columns=cols).to_pandas()
    sep = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(input_path, sep=sep)


class StackedEnsembleInference:
    def __init__(self, models_dir, cpg_csv_path):
        self.models_dir = Path(models_dir)
        cpg_df = pd.read_csv(cpg_csv_path)
        self.cpg_list = cpg_df["CpG"].astype(str).str.strip().dropna().unique().tolist()
        self.cpg_scaler = joblib.load(self.models_dir / "cpg_scaler.joblib")

        base_dir = self.models_dir / "base_models"
        self.base_models = {}
        for name in ("rsf", "gbm", "svm", "weibull"):
            self.base_models[name] = joblib.load(base_dir / f"{name}_full.joblib")
            patch_sklearn_estimator(self.base_models[name])

        mtlr_path = base_dir / "mtlr_full.pt"
        checkpoint = torch.load(mtlr_path, map_location="cpu", weights_only=False)
        cfg = checkpoint["cfg"]
        self.mtlr_cuts = checkpoint["cuts"]
        net = MLPVanilla(
            len(self.cpg_list),
            list(cfg["layers"]),
            cfg["num_durations"],
            cfg["batchnorm"],
            cfg["dropout"],
        )
        self.mtlr_model = PycoxMTLR(net, ttoptim.Adam(lr=cfg["lr"]), duration_index=self.mtlr_cuts)
        self.mtlr_model.net.load_state_dict(checkpoint["state_dict"])

        scalers_dir = self.models_dir / "pred_scalers"
        self.pred_scalers = {
            name: joblib.load(scalers_dir / f"pred_scaler_{name}.joblib")
            for name in ("rsf", "gbm", "svm", "weibull", "mtlr")
        }
        self.stacking_model = load_keras_model_robust(self.models_dir / "stacking_meta_learner.keras")
        self.age_regressor = joblib.load(self.models_dir / "age_regressor.joblib")

    def predict(self, df_meth):
        df_clean = df_meth.copy()
        missing_cpgs = [c for c in self.cpg_list if c not in df_clean.columns]
        if missing_cpgs:
            print(f"[WARNING] Missing {len(missing_cpgs)} of {len(self.cpg_list)} CpGs; setting missing probes to 0.5.")
            for c in missing_cpgs:
                df_clean[c] = 0.5

        X_scaled = self.cpg_scaler.transform(df_clean[self.cpg_list])
        raw_preds = {
            name: pred_risk(self.base_models[name], X_scaled)
            for name in ("rsf", "gbm", "svm", "weibull")
        }
        raw_preds["mtlr"] = mtlr_risk_scores(self.mtlr_model, X_scaled.astype(np.float32), self.mtlr_cuts)

        scaled_preds = {
            name: self.pred_scalers[name].transform(pred.reshape(-1, 1)).ravel()
            for name, pred in raw_preds.items()
        }
        X_meta = np.column_stack([scaled_preds[k] for k in ("rsf", "gbm", "svm", "weibull", "mtlr")])
        ensemble_risk = self.stacking_model.predict(X_meta.astype(np.float32), verbose=0).ravel()
        age_pred = self.age_regressor.predict(ensemble_risk.reshape(-1, 1)).ravel()

        out_df = pd.DataFrame(index=df_meth.index)
        for col in ("Share_ID", "Patient_ID", "Subject_ID"):
            if col in df_meth.columns:
                out_df[col] = df_meth[col]
        if "age" in df_meth.columns:
            out_df["age_true"] = df_meth["age"]
            out_df["age_acceleration"] = age_pred - df_meth["age"]

        out_df["ensemble_risk"] = ensemble_risk
        out_df["calibrated_age_pred"] = age_pred
        for name in ("rsf", "gbm", "svm", "weibull", "mtlr"):
            out_df[f"pred_{name}"] = raw_preds[name]
            out_df[f"pred_{name}_scaled"] = scaled_preds[name]
        return out_df


def main():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Score DNA methylation samples with the pretrained Ensemble Survival Clock."
    )
    parser.add_argument(
        "-i", "--input", required=True, type=Path,
        help="Input methylation table (.csv, .tsv, or .parquet) with one row per sample and CpG beta columns.",
    )
    parser.add_argument(
        "-o", "--output", required=True, type=Path,
        help="Output path (.csv or .parquet) for ensemble risk and rescaled age predictions.",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=repo_root / "pretrained_models",
        help="Directory containing pretrained_models/ (default: ./pretrained_models).",
    )
    parser.add_argument(
        "--cpg-csv", type=Path, default=repo_root / "coxnet_selected_cpgs_190.csv",
        help="Path to coxnet_selected_cpgs_190.csv (default: ./coxnet_selected_cpgs_190.csv).",
    )
    args = parser.parse_args()

    pipeline = StackedEnsembleInference(models_dir=args.models_dir, cpg_csv_path=args.cpg_csv)
    df_in = load_input_table(args.input, pipeline.cpg_list)
    preds = pipeline.predict(df_in)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".parquet":
        preds.to_parquet(args.output, index=False)
    else:
        preds.to_csv(args.output, index=False)
    print(f"Saved predictions for {len(preds)} sample(s) to {args.output}")


if __name__ == "__main__":
    main()
