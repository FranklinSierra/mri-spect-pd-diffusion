from __future__ import annotations

import os
from pathlib import Path
import argparse
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


DATA_INFO_PATH = Path("data_info/data_info.csv")
LATENT_ROOT = Path("result/cond_unet_latents")


def extract_features(volume: np.ndarray) -> np.ndarray:
    """Summarize a (D, H, W) latent volume into a compact 1D feature vector."""
    vol = volume.astype(np.float32, copy=False)
    global_stats = np.array(
        [
            vol.mean(),
            vol.std(),
            vol.min(),
            vol.max(),
            np.median(vol),
            np.percentile(vol, 25),
            np.percentile(vol, 75),
        ],
        dtype=np.float32,
    )
    # Axis-wise averages capture coarse spatial structure without a huge vector.
    axis0 = vol.mean(axis=(1, 2))  # depth profile
    axis1 = vol.mean(axis=(0, 2))  # height profile
    axis2 = vol.mean(axis=(0, 1))  # width profile
    return np.concatenate([global_stats, axis0, axis1, axis2]).astype(np.float32)


def load_split(
    split: str, label_map: Dict[str, int], representation: str, raw_stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    split_dir = LATENT_ROOT / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    features: List[np.ndarray] = []
    labels: List[int] = []
    ids: List[str] = []
    missing_labels: List[str] = []

    for npy_path in sorted(split_dir.glob("*.npy")):
        sample_id = npy_path.stem
        if sample_id not in label_map:
            missing_labels.append(sample_id)
            continue
        volume = np.load(npy_path)
        if representation == "features":
            features.append(extract_features(volume))
        elif representation == "raw":
            vol = volume.astype(np.float32, copy=False)[::raw_stride, ::raw_stride, ::raw_stride]
            features.append(vol.ravel())
        else:
            raise ValueError(f"Unknown representation: {representation}")
        labels.append(label_map[sample_id])
        ids.append(sample_id)

    if not features:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            ids,
            missing_labels,
        )

    return (
        np.vstack(features).astype(np.float32),
        np.array(labels, dtype=np.float32),
        ids,
        missing_labels,
    )


def load_unlabeled_split(
    split_dir: Path, representation: str, raw_stride: int = 1
) -> Tuple[np.ndarray, List[str]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    features: List[np.ndarray] = []
    ids: List[str] = []

    for npy_path in sorted(split_dir.glob("*.npy")):
        sample_id = npy_path.stem
        volume = np.load(npy_path)
        if representation == "features":
            features.append(extract_features(volume))
        elif representation == "raw":
            vol = volume.astype(np.float32, copy=False)[::raw_stride, ::raw_stride, ::raw_stride]
            features.append(vol.ravel())
        else:
            raise ValueError(f"Unknown representation: {representation}")
        ids.append(sample_id)

    if not features:
        return np.empty((0, 0), dtype=np.float32), ids

    return np.vstack(features).astype(np.float32), ids


def standardize(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train - mean) / std, (test - mean) / std


def compute_standardization(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def train_logistic(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 400,
    l2: float = 1e-3,
    eval_data: Optional[Tuple[np.ndarray, np.ndarray, str]] = None,
) -> Tuple[np.ndarray, float]:
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float32)
    bias = 0.0

    for epoch in range(epochs):
        logits = X @ weights + bias
        preds = sigmoid(logits)
        error = preds - y

        grad_w = (X.T @ error) / n_samples + l2 * weights
        grad_b = error.mean()

        weights -= lr * grad_w
        bias -= lr * grad_b

        if (epoch + 1) % 50 == 0:
            loss = -(
                y * np.log(preds + 1e-8) + (1 - y) * np.log(1 - preds + 1e-8)
            ).mean() + 0.5 * l2 * np.sum(weights * weights)
            print(f"Epoch {epoch + 1:04d} | loss {loss:.4f}")

    if eval_data is not None:
        X_eval, y_eval, split_name = eval_data
        if X_eval.size and y_eval.size:
            eval_logits = X_eval @ weights + bias
            eval_probs = sigmoid(eval_logits)
            eval_preds = (eval_probs >= 0.5).astype(np.float32)
            acc = float((eval_preds == y_eval).mean())
            tp = float(((eval_preds == 1) & (y_eval == 1)).sum())
            tn = float(((eval_preds == 0) & (y_eval == 0)).sum())
            fp = float(((eval_preds == 1) & (y_eval == 0)).sum())
            fn = float(((eval_preds == 0) & (y_eval == 1)).sum())
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            # Inline AUC to avoid dependency on evaluate().
            n_pos = float((y_eval == 1).sum())
            n_neg = float((y_eval == 0).sum())
            if n_pos > 0 and n_neg > 0:
                ranks = pd.Series(eval_probs).rank(method="average").to_numpy()
                sum_pos = float(ranks[y_eval == 1].sum())
                auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            else:
                auc = float("nan")
            print(
                f"[{split_name}] after training | acc={acc:.3f} precision={precision:.3f} "
                f"recall={recall:.3f} f1={f1:.3f} auc={auc:.3f} "
                f"(tp={int(tp)}, fp={int(fp)}, fn={int(fn)}, tn={int(tn)})"
            )

    return weights, bias


def train_linear_svm(
    X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 300, l2: float = 1e-3
) -> Tuple[np.ndarray, float]:
    # Uses hinge loss with subgradient descent.
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float32)
    bias = 0.0
    y_pm = 2 * y - 1  # map {0,1} -> {-1,+1}

    for epoch in range(epochs):
        margins = y_pm * (X @ weights + bias)
        active = margins < 1
        if active.any():
            grad_w = l2 * weights - (X[active] * y_pm[active][:, None]).mean(axis=0)
            grad_b = -(y_pm[active]).mean()
        else:
            grad_w = l2 * weights
            grad_b = 0.0

        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights, bias


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    epochs: int = 400,
    l2: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_samples, n_features = X.shape
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((n_features, hidden_dim), dtype=np.float32) * 0.01
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = rng.standard_normal(hidden_dim, dtype=np.float32) * 0.01
    b2 = 0.0

    for epoch in range(epochs):
        # Forward
        h_pre = X @ W1 + b1
        h = np.maximum(h_pre, 0.0)
        logits = h @ W2 + b2
        probs = sigmoid(logits)

        # Loss and gradients
        error = probs - y
        grad_W2 = h.T @ error / n_samples + l2 * W2
        grad_b2 = error.mean()

        grad_h = np.outer(error, W2)
        grad_h[h_pre <= 0] = 0.0
        grad_W1 = X.T @ grad_h / n_samples + l2 * W1
        grad_b1 = grad_h.mean(axis=0)

        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        W2 -= lr * grad_W2
        b2 -= lr * grad_b2

        if (epoch + 1) % 100 == 0:
            loss = -(
                y * np.log(probs + 1e-8) + (1 - y) * np.log(1 - probs + 1e-8)
            ).mean() + 0.5 * l2 * (np.sum(W1 * W1) + np.sum(W2 * W2))
            print(f"MLP epoch {epoch + 1:04d} | loss {loss:.4f}")

    return W1, b1, W2, b2


def knn_predict_proba(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, k: int, leave_one_out: bool = False
) -> np.ndarray:
    y_train = y_train.astype(np.float32)
    probs = np.zeros(len(X_eval), dtype=np.float32)

    for i, x in enumerate(X_eval):
        dists = np.linalg.norm(X_train - x, axis=1)
        if leave_one_out:
            dists[i] = np.inf
        neighbor_idx = np.argpartition(dists, k)[:k]
        probs[i] = y_train[neighbor_idx].mean()
    return probs


def evaluate(split: str, y: np.ndarray, probs: np.ndarray) -> None:
    preds = (probs >= 0.5).astype(np.float32)

    acc = float((preds == y).mean())
    tp = float(((preds == 1) & (y == 1)).sum())
    tn = float(((preds == 0) & (y == 0)).sum())
    fp = float(((preds == 1) & (y == 0)).sum())
    fn = float(((preds == 0) & (y == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    auc = compute_auc(y, probs)

    print(
        f"[{split}] n={len(y)} | acc={acc:.3f} precision={precision:.3f} recall={recall:.3f} "
        f"f1={f1:.3f} auc={auc:.3f} (tp={int(tp)}, fp={int(fp)}, fn={int(fn)}, tn={int(tn)})"
    )


def compute_auc(y: np.ndarray, probs: np.ndarray) -> float:
    # Mann-Whitney U statistic based AUC with average ranks for ties.
    y = y.astype(np.float32)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(probs).rank(method="average").to_numpy()
    sum_pos = float(ranks[y == 1].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def load_latest_nhy(ppmi_csv: Path, ids: set[int]) -> Dict[int, Optional[int]]:
    if not ppmi_csv.exists():
        raise FileNotFoundError(f"PPMI file not found at {ppmi_csv}")

    df = pd.read_csv(ppmi_csv, low_memory=False)
    df = df[df["PATNO"].isin(ids)]
    if df.empty:
        return {}

    df["INFODT_dt"] = pd.to_datetime(df["INFODT"], errors="coerce", format="%m/%Y")
    mask = df["INFODT_dt"].isna()
    if mask.any():
        df.loc[mask, "INFODT_dt"] = pd.to_datetime(df.loc[mask, "INFODT"], errors="coerce")
    # Use very old date for missing to avoid them becoming the latest.
    df["INFODT_dt"] = df["INFODT_dt"].fillna(pd.Timestamp.min)

    df = df.sort_values(["PATNO", "INFODT_dt"])
    latest = df.groupby("PATNO").tail(1)

    stage_map: Dict[int, Optional[int]] = {}
    for _, row in latest.iterrows():
        nhy = row.get("NHY")
        patno = int(row["PATNO"])
        if pd.isna(nhy):
            stage_map[patno] = None
        else:
            # NHY is 0,1,2,...; stage is 1-based for reporting.
            stage_map[patno] = int(float(nhy)) + 1
    return stage_map


def report_stage_probs(
    ids: List[str], probs: np.ndarray, stage_map: Dict[int, Optional[int]], save_path: Optional[Path] = None
) -> None:
    records = []
    for pid_str, prob in zip(ids, probs):
        pid = int(pid_str)
        stage = stage_map.get(pid)
        records.append({"PATNO": pid, "stage": stage, "prob_parkinson": float(prob)})

    df = pd.DataFrame(records)
    grouped = df.groupby("stage")["prob_parkinson"]
    summary = grouped.agg(["count", "mean", "median"]).reset_index().sort_values("stage", na_position="last")

    print("Probabilidades de test agrupadas por estadio (NHY+1):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Probabilidades individuales de test guardadas en {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate classifiers on latent volumes.")
    parser.add_argument(
        "--representation",
        choices=["features", "raw"],
        default="features",
        help="Use compact statistical features or raw flattened volumes.",
    )
    parser.add_argument(
        "--classifier",
        choices=["logistic", "svm", "knn", "mlp"],
        default="logistic",
        help="Classifier to train.",
    )
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs for gradient-based models.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for gradient-based models.")
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 weight decay/regularization.")
    parser.add_argument("--k", type=int, default=5, help="Neighbors for KNN.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden size for MLP.")
    parser.add_argument(
        "--raw-stride",
        type=int,
        default=1,
        help="Stride for subsampling raw volumes (use >1 to downsample and reduce dimensionality).",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Ruta opcional para guardar pesos y bias del modelo entrenado junto a la estandarización.",
    )
    parser.add_argument(
        "--save-test-probs",
        type=Path,
        default=None,
        help="Ruta opcional para guardar probabilidades de test por ID y estadio (si se provee PPMI).",
    )
    parser.add_argument(
        "--save-prodromal-probs",
        type=Path,
        default=None,
        help="Ruta opcional para guardar probabilidades de prodromal por ID.",
    )
    parser.add_argument(
        "--prodromal-dir",
        type=Path,
        default=LATENT_ROOT / "prodromal",
        help="Directorio con latentes prodromales (.npy) para inferencia.",
    )
    parser.add_argument(
        "--eval-splits",
        type=str,
        nargs="+",
        choices=["test", "prodromal"],
        default=["test", "prodromal"],
        help="Qué splits evaluar/inferir después de entrenar.",
    )
    parser.add_argument(
        "--ppmi-csv",
        type=Path,
        default=Path("PPMI_MASTER_v2.csv"),
        help="Archivo maestro con la columna NHY para agrupar por estadio.",
    )
    parser.add_argument(
        "--test-id-file",
        type=Path,
        default=Path("data_info/test.txt"),
        help="Archivo con IDs de test (PATNO) para cruzar con PPMI y obtener NHY.",
    )
    args = parser.parse_args()

    if not DATA_INFO_PATH.exists():
        raise FileNotFoundError(f"Missing labels CSV at {DATA_INFO_PATH}")

    df = pd.read_csv(DATA_INFO_PATH, dtype={"ID": str, "label": np.int32})
    label_map = dict(zip(df["ID"].astype(str), df["label"].astype(int)))

    X_train, y_train, ids_train, missing_train = load_split(
        "train", label_map, args.representation, raw_stride=args.raw_stride
    )
    eval_test = "test" in args.eval_splits
    eval_prodromal = "prodromal" in args.eval_splits

    if eval_test:
        X_test, y_test, ids_test, missing_test = load_split(
            "test", label_map, args.representation, raw_stride=args.raw_stride
        )
    else:
        X_test, y_test, ids_test, missing_test = (
            np.empty((0, 0), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
            [],
        )

    # Prodromal es un split adicional sin labels (opcional).
    prodromal_dir = args.prodromal_dir
    X_prodromal: np.ndarray
    ids_prodromal: List[str]
    if eval_prodromal and prodromal_dir.exists():
        X_prodromal, ids_prodromal = load_unlabeled_split(
            prodromal_dir, args.representation, raw_stride=args.raw_stride
        )
        print(f"Loaded {len(ids_prodromal)} prodromal samples.")
    else:
        X_prodromal, ids_prodromal = np.empty((0, 0), dtype=np.float32), []

    if missing_train:
        print(f"Warning: {len(missing_train)} train samples missing labels: {missing_train[:5]}...")
    if missing_test:
        print(f"Warning: {len(missing_test)} test samples missing labels: {missing_test[:5]}...")

    if X_train.size == 0:
        raise RuntimeError("No training samples were loaded. Check paths and labels.")

    train_mean, train_std = compute_standardization(X_train)
    X_train_std = (X_train - train_mean) / train_std
    X_test_std = (X_test - train_mean) / train_std if X_test.size else X_test
    X_prodromal_std = (X_prodromal - train_mean) / train_std if X_prodromal.size else X_prodromal

    print(
        f"Loaded {len(X_train)} train samples with {X_train.shape[1]} features each "
        f"(representation={args.representation}, classifier={args.classifier})."
    )
    if eval_test:
        print(f"Loaded {len(X_test)} test samples.")
    else:
        print("Test evaluation disabled by --eval-splits.")
    if eval_prodromal and ids_prodromal:
        print(f"Loaded {len(ids_prodromal)} prodromal samples.")
    elif eval_prodromal:
        print(f"No prodromal samples found at {args.prodromal_dir}.")
    else:
        print("Prodromal inference disabled by --eval-splits.")

    prodromal_probs = np.array([])

    if args.classifier == "logistic":
        eval_tuple = None
        if eval_test and X_test_std.size:
            eval_tuple = (X_test_std, y_test, "test")
        weights, bias = train_logistic(
            X_train_std, y_train, lr=args.lr, epochs=args.epochs, l2=args.l2, eval_data=eval_tuple
        )
        train_probs = sigmoid(X_train_std @ weights + bias)
        test_probs = sigmoid(X_test_std @ weights + bias) if X_test_std.size else np.array([])
        prodromal_probs = sigmoid(X_prodromal_std @ weights + bias) if X_prodromal_std.size else np.array([])
    elif args.classifier == "svm":
        weights, bias = train_linear_svm(X_train_std, y_train, lr=args.lr, epochs=args.epochs, l2=args.l2)
        train_probs = sigmoid(X_train_std @ weights + bias)
        test_probs = sigmoid(X_test_std @ weights + bias) if X_test_std.size else np.array([])
        prodromal_probs = sigmoid(X_prodromal_std @ weights + bias) if X_prodromal_std.size else np.array([])
    elif args.classifier == "knn":
        train_probs = knn_predict_proba(X_train_std, y_train, X_train_std, args.k, leave_one_out=True)
        test_probs = knn_predict_proba(X_train_std, y_train, X_test_std, args.k) if X_test_std.size else np.array([])
        prodromal_probs = knn_predict_proba(X_train_std, y_train, X_prodromal_std, args.k) if X_prodromal_std.size else np.array([])
    elif args.classifier == "mlp":
        W1, b1, W2, b2 = train_mlp(
            X_train_std, y_train, hidden_dim=args.hidden_dim, lr=args.lr, epochs=args.epochs, l2=args.l2
        )
        def _forward(X: np.ndarray) -> np.ndarray:
            h = np.maximum(X @ W1 + b1, 0.0)
            return sigmoid(h @ W2 + b2)
        train_probs = _forward(X_train_std)
        test_probs = _forward(X_test_std) if X_test_std.size else np.array([])
        prodromal_probs = _forward(X_prodromal_std) if X_prodromal_std.size else np.array([])
    else:
        raise ValueError(f"Unsupported classifier: {args.classifier}")

    if args.save_model:
        model_payload = {
            "classifier": args.classifier,
            "representation": args.representation,
            "train_mean": train_mean,
            "train_std": train_std,
        }
        if args.classifier in {"logistic", "svm"}:
            model_payload["weights"] = weights
            model_payload["bias"] = bias
        elif args.classifier == "mlp":
            model_payload.update({"W1": W1, "b1": b1, "W2": W2, "b2": b2})
        elif args.classifier == "knn":
            model_payload.update({"X_train": X_train_std, "y_train": y_train, "k": args.k})
        np.savez(args.save_model, **model_payload)
        print(f"Modelo guardado en {args.save_model}")

    evaluate("train", y_train, train_probs)
    if eval_test and X_test_std.size:
        evaluate("test", y_test, test_probs)
        if args.ppmi_csv.exists():
            stage_map = load_latest_nhy(args.ppmi_csv, set(map(int, ids_test)))
            report_stage_probs(ids_test, test_probs, stage_map, save_path=args.save_test_probs)
        else:
            print(f"Archivo PPMI no encontrado en {args.ppmi_csv}, se omite agrupación por estadio.")
    elif eval_test:
        print("No test samples to evaluate.")
    else:
        print("Test evaluation skipped (--eval-splits).")

    if eval_prodromal and X_prodromal_std.size:
        prodromal_df = pd.DataFrame(
            {"id": ids_prodromal, "prob_parkinson": prodromal_probs.astype(float)}
        )
        save_path = args.save_prodromal_probs or Path("prodromal_probs.csv")
        prodromal_df.to_csv(save_path, index=False)
        print(f"Probabilidades prodromales guardadas en {save_path}")
    elif eval_prodromal:
        print("Prodromal inference requested but no samples found.")


if __name__ == "__main__":
    main()
