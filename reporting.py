import os
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch


def ensure_dir(path):
    path = str(path)
    if path:
        os.makedirs(path, exist_ok=True)


def save_json_report(data: Dict[str, Any], path: str) -> str:
    dirname = os.path.dirname(str(path))
    ensure_dir(dirname)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    return str(path)


def count_parameters(model: torch.nn.Module) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    by_module = {}
    for name, child in model.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        if n_params > 0:
            by_module[name] = int(n_params)

    buffer_numel = sum(b.numel() for b in model.buffers())

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(total_params - trainable_params),
        "buffer_numel": int(buffer_numel),
        "estimated_model_size_mb": round(total_params * 4 / (1024 * 1024), 3),
        "params_by_module": by_module,
    }

def move_batch(batch, device):
    motif_emb, motif_mask, lstm_ids, lstm_mask, doc_word_freq, labels = batch

    return (
        motif_emb.to(device),
        motif_mask.to(device),
        lstm_ids.to(device),
        lstm_mask.to(device),
        doc_word_freq.to(device),
        labels.to(device),
    )


@torch.no_grad()
def benchmark_inference(
    model: torch.nn.Module,
    loader,
    device: str,
    warmup_batches: int = 1,
) -> Dict[str, float]:

    model.eval()

    use_cuda = (
        isinstance(device, str)
        and device.startswith("cuda")
        and torch.cuda.is_available()
    )

    if len(loader) == 0:
        return {
            "total_inference_time_s": 0.0,
            "n_samples": 0,
            "n_batches": 0,
            "avg_ms_per_sample": 0.0,
            "avg_ms_per_batch": 0.0,
            "samples_per_sec": 0.0,
        }

    for i, batch in enumerate(loader):
        if i >= warmup_batches:
            break

        (
            motif_emb,
            motif_mask,
            lstm_ids,
            lstm_mask,
            doc_word_freq,
            labels,
        ) = move_batch(batch, device)

        _ = model(
            motif_emb,
            motif_mask,
            lstm_ids,
            lstm_mask,
            doc_word_freq,
        )

    total_samples = 0
    total_batches = 0
    total_time = 0.0
    batch_times = []

    for batch in loader:
        (
            motif_emb,
            motif_mask,
            lstm_ids,
            lstm_mask,
            doc_word_freq,
            labels,
        ) = move_batch(batch, device)

        if use_cuda:
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        _ = model(
            motif_emb,
            motif_mask,
            lstm_ids,
            lstm_mask,
            doc_word_freq,
        )

        if use_cuda:
            torch.cuda.synchronize()

        dt = time.perf_counter() - t0

        total_time += dt
        batch_times.append(dt)
        total_batches += 1
        total_samples += labels.size(0)

    total_time = float(total_time)
    total_samples = int(total_samples)
    total_batches = int(total_batches)

    return {
        "total_inference_time_s": total_time,
        "n_samples": total_samples,
        "n_batches": total_batches,
        "avg_ms_per_sample": float(
            (total_time / max(total_samples, 1)) * 1000.0
        ),
        "avg_ms_per_batch": float(
            (total_time / max(total_batches, 1)) * 1000.0
        ),
        "samples_per_sec": float(
            total_samples / max(total_time, 1e-8)
        ),
        "warmup_batches": warmup_batches,
    }


def save_fold_predictions(
    out_dir: str,
    fold: int,
    global_idx: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    extra_arrays: Optional[Dict[str, np.ndarray]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:

    ensure_dir(out_dir)

    fold = int(fold)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    global_idx = np.asarray(global_idx).astype(int)

    base_name = f"fold_{fold:02d}_test_predictions"
    base_path = os.path.join(str(out_dir), base_name)

    npz_kwargs = {
        "fold": np.array(fold),
        "global_idx": global_idx,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "threshold": np.array(threshold),
    }

    if extra_arrays is not None:
        for k, v in extra_arrays.items():
            npz_kwargs[k] = np.asarray(v)

    np.savez_compressed(base_path + ".npz", **npz_kwargs)

    df = pd.DataFrame(
        {
            "fold": fold,
            "global_idx": global_idx,
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }
    )

    df.to_csv(base_path + ".csv", index=False)

    metadata = {
        "fold": fold,
        "threshold": threshold,
        "n_samples": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "saved_at": datetime.now().isoformat(),
    }

    if extra_metadata is not None:
        metadata.update(extra_metadata)

    save_json_report(metadata, base_path + "_meta.json")

    return df


def summarize_fold_reports(fold_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create numeric summary over all fold reports.
    """
    if len(fold_reports) == 0:
        return {
            "meta": {},
            "numeric": {},
        }

    df = pd.json_normalize(fold_reports, sep=".")
    numeric_df = df.select_dtypes(include=[np.number])

    numeric_summary = {}

    for col in numeric_df.columns:
        values = numeric_df[col].dropna()

        if len(values) == 0:
            continue

        numeric_summary[col] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }

    meta_summary = {
        "n_folds": len(fold_reports),
    }

    important_keys = [
        "test_metrics.auc",
        "test_metrics.accuracy",
        "test_metrics.sensitivity",
        "test_metrics.specificity",
        "test_metrics.f1",
        "training.total_train_time_s",
        "training.avg_epoch_time_s",
        "inference.avg_ms_per_sample",
        "params.total_params",
    ]

    for key in important_keys:
        if key in numeric_df.columns:
            meta_summary[f"mean_{key}"] = float(numeric_df[key].mean())
            if len(numeric_df) > 1:
                meta_summary[f"std_{key}"] = float(numeric_df[key].std(ddof=1))
            else:
                meta_summary[f"std_{key}"] = 0.0

    return {
        "meta": meta_summary,
        "numeric": numeric_summary,
    }


def save_overall_report(
    report_dir: str,
    run_config: Dict[str, Any],
    fold_reports: List[Dict[str, Any]],
    prediction_summary: Optional[Dict[str, Any]] = None,
    tag: str = "",
) -> tuple:
    """
    Save overall model/reporting output:
        - JSON full report
        - Excel report
    """
    ensure_dir(report_dir)

    summary = summarize_fold_reports(fold_reports)

    overall_report = {
        "generated_at": datetime.now().isoformat(),
        "run_config": run_config,
        "prediction_summary": prediction_summary,
        "summary": summary,
        "fold_reports": fold_reports,
    }

    json_path = os.path.join(str(report_dir), f"overall_model_report{tag}.json")
    save_json_report(overall_report, json_path)

    xlsx_path = os.path.join(str(report_dir), f"overall_model_report{tag}.xlsx")

    fold_df = pd.json_normalize(fold_reports, sep=".") if fold_reports else pd.DataFrame()
    config_df = pd.json_normalize(run_config, sep=".") if run_config else pd.DataFrame()

    summary_numeric_df = pd.DataFrame.from_dict(summary["numeric"], orient="index")
    summary_meta_df = pd.DataFrame([summary["meta"]])

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:

        if not config_df.empty:
            config_df.T.to_excel(writer, sheet_name="config", header=False)

        if not fold_df.empty:
            fold_df.to_excel(writer, sheet_name="fold_reports", index=False)

        if not summary_numeric_df.empty:
            summary_numeric_df.to_excel(writer, sheet_name="numeric_summary")

        summary_meta_df.to_excel(writer, sheet_name="meta_summary", index=False)

        if prediction_summary is not None:
            pd.DataFrame([prediction_summary]).to_excel(
                writer,
                sheet_name="prediction_summary",
                index=False,
            )

    return json_path, xlsx_path