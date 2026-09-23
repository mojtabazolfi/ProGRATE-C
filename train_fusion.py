import os
import sys
import time
import argparse
import functools
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import roc_curve

from config import DEVICE, CHECKPOINT_DIR, LOG_DIR, FIGURES_DIR, TRAIN_CONFIG, EMBEDDINGS_DIR, PREDICTIONS_DIR, REPORT_DIR
from utils import (
    set_global_seed, compute_metrics, format_metrics_table,
    build_scheduler, save_json, FocalLoss,
)
from dataloader_fusion import prepare_cv_folds, build_fold_data, fusion_collate_fn
from model_fusion import build_fusion_model
from cv_report import (
    save_epoch_csv, save_test_summary_excel,
    plot_loss_curves, plot_accuracy_curves, plot_mean_roc,
)
import pandas as pd

from reporting import count_parameters, benchmark_inference, save_fold_predictions, save_json_report, save_overall_report

def parse_args():
    parser = argparse.ArgumentParser(description="MGM-Former + SIGD Fusion Training")
    parser.add_argument("--data", type=str, nargs="+", required=False,
                    default=["E:\\INSF\\motif\\data\\raw\\all_data_32.tsv"],
                    help="One or more .tsv files in the fusion format")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Kept small since the GCN graph is rebuilt every batch")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-splits", type=int, default=10,
                        help="Number of stratified CV folds")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--test-mode", action="store_true",
                        help="Quick sanity run: 3 epochs")
    return parser.parse_args()


def make_loader(dataset, vocab_size, batch_size, shuffle):
    collate = functools.partial(fusion_collate_fn, vocab_size=vocab_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       collate_fn=collate, num_workers=0, drop_last=False)


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_predictions=False):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_proba, all_labels = [], [], []

    for motif_emb, motif_mask, lstm_ids, lstm_mask, doc_word_freq, labels in loader:
        motif_emb = motif_emb.to(device)
        motif_mask = motif_mask.to(device)
        lstm_ids = lstm_ids.to(device)
        lstm_mask = lstm_mask.to(device)
        doc_word_freq = doc_word_freq.to(device)
        labels = labels.to(device)

        logits, _ = model(motif_emb, motif_mask, lstm_ids, lstm_mask, doc_word_freq)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.softmax(logits.float(), dim=1)
        probs = torch.nan_to_num(probs, nan=0.5)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_proba.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(np.array(all_labels), np.array(all_proba))
    metrics["loss"] = avg_loss
    metrics["accuracy"] = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    if return_predictions:
        return metrics, np.array(all_labels), np.array(all_proba)
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip):
    model.train()
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []

    for motif_emb, motif_mask, lstm_ids, lstm_mask, doc_word_freq, labels in loader:
        motif_emb = motif_emb.to(device)
        motif_mask = motif_mask.to(device)
        lstm_ids = lstm_ids.to(device)
        lstm_mask = lstm_mask.to(device)
        doc_word_freq = doc_word_freq.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(motif_emb, motif_mask, lstm_ids, lstm_mask, doc_word_freq)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        preds = torch.argmax(logits, dim=1).detach()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(n_batches, 1)
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    return {"loss": avg_loss, "accuracy": acc}


def run_fold(
    fold_num,
    n_folds,
    records,
    labels,
    motif_emb_full,
    trainval_idx,
    test_idx,
    args,
    device,
    tag_suffix,
    pred_dir,
    report_dir,
):
    fold_data = build_fold_data(
        records,
        labels,
        motif_emb_full,
        trainval_idx,
        test_idx,
        seed=42,
        fold_num=fold_num,
    )

    datasets = fold_data["datasets"]
    word_id_map = fold_data["word_id_map"]
    graph_stats = fold_data["graph_stats"]
    vocab_size = len(word_id_map)

    train_loader = make_loader(datasets["train"], vocab_size, args.batch_size, shuffle=True)
    val_loader = make_loader(datasets["val"], vocab_size, args.batch_size, shuffle=False)
    test_loader = make_loader(datasets["test"], vocab_size, args.batch_size, shuffle=False)

    model = build_fusion_model(vocab_size, graph_stats, num_classes=2, device=device)

    param_summary = count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    TRAIN_CONFIG.max_epochs = args.epochs
    scheduler = build_scheduler(
        optimizer,
        scheduler_name="cosine_warmup",
        max_epochs=args.epochs,
    )

    ckpt_path = os.path.join(
        str(CHECKPOINT_DIR),
        f"fusion_model{tag_suffix}_fold{fold_num}.pt",
    )

    best_val_loss = float("inf")
    best_epoch = -1

    hist = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
    }

    epoch_times = []

    print(
        f"\nFold {fold_num}/{n_folds}: training for {args.epochs} epochs "
        f"(batch_size={args.batch_size}, lr={args.lr})..."
    )

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    fold_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        model.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            args.grad_clip,
        )

        val_stats = evaluate(model, val_loader, criterion, device)

        hist["train_loss"].append(train_stats["loss"])
        hist["train_acc"].append(train_stats["accuracy"])
        hist["val_loss"].append(val_stats["loss"])
        hist["val_acc"].append(val_stats["accuracy"])
        hist["val_auc"].append(val_stats.get("auc", 0.0))

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_score": best_val_loss,
                    "best_epoch": best_epoch,
                },
                ckpt_path,
            )

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(
            f"  [fold {fold_num}] Epoch {epoch + 1:3d}/{args.epochs} ({epoch_time:.1f}s) | "
            f"train_loss={train_stats['loss']:.4f}, train_acc={train_stats['accuracy']:.4f} | "
            f"val_loss={val_stats['loss']:.4f}, val_acc={val_stats['accuracy']:.4f}, "
            f"val_auc={val_stats.get('auc', 0):.4f}"
        )

    fold_time = time.time() - fold_start
    avg_epoch_time = float(np.mean(epoch_times)) if len(epoch_times) > 0 else 0.0

    print(
        f"\n  Fold {fold_num} training complete in {fold_time / 60:.2f} min "
        f"(best val_loss={best_val_loss:.4f} at epoch {best_epoch + 1})"
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\n  Fold {fold_num} TEST SET EVALUATION (best-val-loss checkpoint)")

    test_metrics, test_labels, test_proba = evaluate(
        model,
        test_loader,
        criterion,
        device,
        return_predictions=True,
    )

    print(format_metrics_table(test_metrics))

    fpr, tpr, _ = roc_curve(test_labels, test_proba)

    inference_stats = benchmark_inference(
        model=model,
        loader=test_loader,
        device=device,
        warmup_batches=1,
    )

    pred_df = save_fold_predictions(
        out_dir=pred_dir,
        fold=fold_num,
        global_idx=np.asarray(test_idx),
        y_true=test_labels,
        y_prob=test_proba,
        threshold=0.5,
        extra_arrays={
            "fpr": fpr,
            "tpr": tpr,
        },
        extra_metadata={
            "auc": test_metrics.get("auc", None),
            "best_epoch": best_epoch + 1,
            "best_val_loss": float(best_val_loss),
            "vocab_size": vocab_size,
        },
    )

    cuda_peak_memory_mb = None
    if str(device).startswith("cuda") and torch.cuda.is_available():
        cuda_peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    fold_report = {
        "fold": fold_num,
        "data": {
            "train_size": len(datasets["train"]),
            "val_size": len(datasets["val"]),
            "test_size": len(datasets["test"]),
            "vocab_size": vocab_size,
        },
        "graph_stats_shapes": {
            k: tuple(np.asarray(v).shape) for k, v in graph_stats.items()
        },
        "params": param_summary,
        "training": {
            "epochs": args.epochs,
            "best_epoch": best_epoch + 1,
            "best_val_loss": float(best_val_loss),
            "total_train_time_s": float(fold_time),
            "total_train_time_min": float(fold_time / 60.0),
            "avg_epoch_time_s": avg_epoch_time,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
        },
        "test_metrics": test_metrics,
        "inference": inference_stats,
        "system": {
            "device": str(device),
            "cuda_peak_memory_mb": cuda_peak_memory_mb,
        },
        "checkpoint": ckpt_path,
        "prediction_file_csv": os.path.join(str(pred_dir), f"fold_{fold_num:02d}_test_predictions.csv"),
        "prediction_file_npz": os.path.join(str(pred_dir), f"fold_{fold_num:02d}_test_predictions.npz"),
    }

    save_json_report(
        fold_report,
        os.path.join(str(report_dir), f"fold_{fold_num:02d}_report{tag_suffix}.json"),
    )

    return (
        hist,
        test_metrics,
        (fpr, tpr, test_metrics["auc"]),
        fold_report,
        pred_df,
    )


def main():
    args = parse_args()
    if args.test_mode:
        args.epochs = 3
        args.n_splits = 3
        print("\n⚠ TEST MODE: 3 epochs, 3 folds\n")

    set_global_seed(42)

    print("=" * 60)
    print(f"  FUSION PIPELINE: MGM-Former + SIGD — {args.n_splits}-fold Stratified CV")
    print("=" * 60)

    cache_dir = str(EMBEDDINGS_DIR)
    cv_data = prepare_cv_folds(
        args.data, cache_dir=cache_dir, n_splits=args.n_splits,
        force_recompute_embeddings=args.force_embeddings,
    )
    records = cv_data["records"]
    labels = cv_data["labels"]
    motif_emb_full = cv_data["motif_emb_full"]
    fold_indices = cv_data["fold_indices"]

    tag_suffix = f"_{args.tag}" if args.tag else ""

    run_name = args.tag if args.tag else "default"
    pred_dir = PREDICTIONS_DIR / run_name
    report_dir = REPORT_DIR / run_name

    pred_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    fold_histories = []
    fold_test_metrics = []
    fold_roc_data = []
    fold_reports = []
    all_pred_dfs = []

    overall_start = time.time()

    for fold_num, (trainval_idx, test_idx) in enumerate(fold_indices, start=1):
        print("\n" + "=" * 60)
        print(f"  FOLD {fold_num}/{len(fold_indices)}")
        print("=" * 60)

        hist, test_metrics, roc_data, fold_report, pred_df = run_fold(
            fold_num,
            len(fold_indices),
            records,
            labels,
            motif_emb_full,
            trainval_idx,
            test_idx,
            args,
            DEVICE,
            tag_suffix,
            pred_dir,
            report_dir,
        )

        fold_histories.append(hist)
        fold_test_metrics.append(test_metrics)
        fold_roc_data.append(roc_data)
        fold_reports.append(fold_report)
        all_pred_dfs.append(pred_df)

    total_time = time.time() - overall_start

    print(f"\n{args.n_splits}-fold CV complete in {total_time / 60:.2f} minutes")

    all_csv = None
    all_npz = None
    prediction_summary = {}

    if len(all_pred_dfs) > 0:
        all_preds = pd.concat(all_pred_dfs, ignore_index=True)

        all_csv = pred_dir / f"all_folds_test_predictions{tag_suffix}.csv"
        all_npz = pred_dir / f"all_folds_test_predictions{tag_suffix}.npz"

        all_preds.to_csv(all_csv, index=False)

        np.savez_compressed(
            all_npz,
            fold=all_preds["fold"].to_numpy(),
            global_idx=all_preds["global_idx"].to_numpy(),
            y_true=all_preds["y_true"].to_numpy(),
            y_prob=all_preds["y_prob"].to_numpy(),
            y_pred=all_preds["y_pred"].to_numpy(),
        )

        prediction_summary = {
            "n_saved_predictions": int(len(all_preds)),
            "n_folds": int(all_preds["fold"].nunique()),
            "n_positive": int((all_preds["y_true"] == 1).sum()),
            "n_negative": int((all_preds["y_true"] == 0).sum()),
            "prediction_csv": str(all_csv),
            "prediction_npz": str(all_npz),
        }

        print(f"\n  Saved aggregated predictions:")
        print(f"    CSV: {all_csv}")
        print(f"    NPZ: {all_npz}")
    

    os.makedirs(str(LOG_DIR), exist_ok=True)

    epoch_csv_path = os.path.join(str(LOG_DIR), f"cv_epoch_metrics{tag_suffix}.csv")
    save_epoch_csv(fold_histories, epoch_csv_path)

    summary_xlsx_path = os.path.join(str(LOG_DIR), f"cv_test_summary{tag_suffix}.xlsx")
    save_test_summary_excel(fold_test_metrics, summary_xlsx_path)

    fig_dir = str(FIGURES_DIR)
    plot_loss_curves(fold_histories, fig_dir, tag=tag_suffix)
    plot_accuracy_curves(fold_histories, fig_dir, tag=tag_suffix)
    roc_summary = plot_mean_roc(fold_roc_data, fig_dir, tag=tag_suffix)

    run_config_path = os.path.join(str(LOG_DIR), f"cv_run_config{tag_suffix}.json")
    save_json({
        "config": vars(args),
        "n_splits": args.n_splits,
        "mean_test_auc": roc_summary["mean_auc"],
        "std_test_auc": roc_summary["std_auc"],
    }, run_config_path)

    os.makedirs(str(LOG_DIR), exist_ok=True)

    epoch_csv_path = os.path.join(str(LOG_DIR), f"cv_epoch_metrics{tag_suffix}.csv")
    save_epoch_csv(fold_histories, epoch_csv_path)

    summary_xlsx_path = os.path.join(str(LOG_DIR), f"cv_test_summary{tag_suffix}.xlsx")
    save_test_summary_excel(fold_test_metrics, summary_xlsx_path)

    fig_dir = str(FIGURES_DIR)

    plot_loss_curves(fold_histories, fig_dir, tag=tag_suffix)
    plot_accuracy_curves(fold_histories, fig_dir, tag=tag_suffix)

    roc_summary = plot_mean_roc(fold_roc_data, fig_dir, tag=tag_suffix)

    run_config = vars(args).copy()

    run_config.update(
        {
            "device": DEVICE,
            "n_samples": int(len(labels)),
            "n_cancer": int((labels == 1).sum()),
            "n_control": int((labels == 0).sum()),
            "total_cv_time_s": float(total_time),
            "total_cv_time_min": float(total_time / 60.0),
            "mean_test_auc": roc_summary["mean_auc"],
            "std_test_auc": roc_summary["std_auc"],
            "prediction_csv": str(all_csv) if all_csv is not None else None,
            "prediction_npz": str(all_npz) if all_npz is not None else None,
        }
    )

    report_json, report_xlsx = save_overall_report(
        report_dir=str(report_dir),
        run_config=run_config,
        fold_reports=fold_reports,
        prediction_summary=prediction_summary,
        tag=tag_suffix,
    )

    print(f"\n  Independent model report JSON: {report_json}")
    print(f"  Independent model report XLSX: {report_xlsx}")


    print(f"\n  Per-epoch metrics CSV:  {epoch_csv_path}")
    print(f"  Test summary Excel:     {summary_xlsx_path}")
    print(f"  Run config JSON:        {run_config_path}")
    print(f"  Figures saved to:       {fig_dir}")
    print(f"\n  Mean test AUC: {roc_summary['mean_auc']:.4f} +/- {roc_summary['std_auc']:.4f}")
    print("\n✓ 10-fold CV fusion training complete!")


if __name__ == "__main__":
    main()
