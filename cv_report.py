import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "text.color": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 16,
    "axes.labelweight": "bold"
})

def _style_axes(ax):
    ax.grid(False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(12)
        
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15

def _save_fig(fig, out_dir, filename_stem):
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{filename_stem}.png"),
                facecolor="white", dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{filename_stem}.pdf"),
                facecolor="white", bbox_inches="tight")
    plt.close(fig)

def save_epoch_csv(fold_histories: list, path: str):
    rows = []
    for fold_num, hist in enumerate(fold_histories, start=1):
        n_epochs = len(hist["train_loss"])
        for e in range(n_epochs):
            rows.append({
                "fold": fold_num,
                "epoch": e + 1,
                "train_loss": hist["train_loss"][e],
                "train_acc": hist["train_acc"][e],
                "val_loss": hist["val_loss"][e],
                "val_acc": hist["val_acc"][e],
                "val_auc": hist["val_auc"][e],
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Per-epoch CV metrics saved to: {path}")
    return df

def save_test_summary_excel(fold_test_metrics: list, path: str):
    display_keys = [
        "accuracy", "balanced_accuracy", "sensitivity", "specificity",
        "precision", "f1", "f1_macro", "mcc", "auc", "auprc",
    ]
    df = pd.DataFrame(fold_test_metrics)[display_keys]
    df.insert(0, "fold", range(1, len(df) + 1))

    mean_row = df[display_keys].mean()
    std_row = df[display_keys].std(ddof=1)

    summary = df.copy()
    summary.loc[len(summary)] = ["mean"] + mean_row.tolist()
    summary.loc[len(summary)] = ["std"] + std_row.tolist()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="test_metrics_per_fold", index=False,
                          float_format="%.4f")

    print(f"  Test-set CV summary (mean +/- std over {len(df)} folds) saved to: {path}")
    print(summary.to_string(index=False))
    return summary


def plot_loss_curves(fold_histories: list, out_dir: str, tag: str = "",
                      dataset_name: str = "Fusion"):
    train_loss = np.array([h["train_loss"] for h in fold_histories])
    val_loss = np.array([h["val_loss"] for h in fold_histories])
    epochs = np.arange(1, train_loss.shape[1] + 1)

    tr_mean, tr_std = train_loss.mean(axis=0), train_loss.std(axis=0, ddof=1)
    va_mean, va_std = val_loss.mean(axis=0), val_loss.std(axis=0, ddof=1)

    best_epochs = [np.argmin(h["val_loss"]) + 1 for h in fold_histories]
    mean_best_epoch = int(np.mean(best_epochs))

    epochs = epochs[:mean_best_epoch]
    tr_mean, tr_std = tr_mean[:mean_best_epoch], tr_std[:mean_best_epoch]
    va_mean, va_std = va_mean[:mean_best_epoch], va_std[:mean_best_epoch]

    c_train = '#0072B2'
    c_val = '#D55E00'

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    
    ax.plot(epochs, tr_mean, color=c_train, lw=2.5, linestyle='-', label="Train (mean)")
    ax.fill_between(epochs, tr_mean - tr_std, tr_mean + tr_std, color=c_train, alpha=0.2)
    ax.plot(epochs, va_mean, color=c_val, lw=2.5, linestyle='--', label="Validation (mean)")
    ax.fill_between(epochs, va_mean - va_std, va_mean + va_std, color=c_val, alpha=0.2)


    ax.set_title(f"ProGRATE end-to-end training loss curve", fontweight="bold", fontsize=16, pad=15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", frameon=False, prop={'weight':'bold', 'size':11})
    
    _style_axes(ax)
    _save_fig(fig, out_dir, f"loss_curves{tag}")

def plot_accuracy_curves(fold_histories: list, out_dir: str, tag: str = "", dataset_name: str = "Fusion"):
    train_acc = np.array([h["train_acc"] for h in fold_histories])
    val_acc = np.array([h["val_acc"] for h in fold_histories])
    epochs = np.arange(1, train_acc.shape[1] + 1)

    tr_mean, tr_std = train_acc.mean(axis=0), train_acc.std(axis=0, ddof=1)
    va_mean, va_std = val_acc.mean(axis=0), val_acc.std(axis=0, ddof=1)

    c_train = '#0072B2' 
    c_val = '#D55E00'   

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    
    ax.plot(epochs, tr_mean, color=c_train, lw=2.5, linestyle='-', label="Train (mean)")
    ax.fill_between(epochs, tr_mean - tr_std, tr_mean + tr_std, color=c_train, alpha=0.2)
    ax.plot(epochs, va_mean, color=c_val, lw=2.5, linestyle='--', label="Validation (mean)")
    ax.fill_between(epochs, va_mean - va_std, va_mean + va_std, color=c_val, alpha=0.2)
    
    min_y = min((tr_mean - tr_std).min(), (va_mean - va_std).min())
    max_y = max((tr_mean + tr_std).max(), (va_mean + va_std).max())
    padding = (max_y - min_y) * 0.1 if (max_y - min_y) > 0 else 0.05
    ax.set_ylim([max(0.0, min_y - padding), min(1.05, max_y + padding)])

    ax.set_title(f"ProGRATE end-to-end training accuracy curve", fontweight="bold", fontsize=16, pad=15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right", frameon=False, prop={'weight':'bold', 'size':11})
    
    _style_axes(ax)
    _save_fig(fig, out_dir, f"acc_curves{tag}")

def plot_mean_roc(fold_fpr_tpr_auc: list, out_dir: str, tag: str = "",
                   dataset_name: str = "Fusion"):
    mean_fpr = np.linspace(0, 1, 200)
    interp_tprs = []
    fold_aucs = []

    for fpr, tpr, auc in fold_fpr_tpr_auc:
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
        fold_aucs.append(auc)

    interp_tprs = np.array(interp_tprs)
    mean_tpr = interp_tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = interp_tprs.std(axis=0, ddof=1)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1.0)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0.0)

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs, ddof=1))

    c_roc = '#0072B2' 

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")

    for fpr, tpr, _ in fold_fpr_tpr_auc:
        ax.plot(fpr, tpr, color="gray", lw=0.8, alpha=0.35)

    ax.plot(mean_fpr, mean_tpr, color=c_roc, lw=2.5, linestyle='-',
            label=f"Mean ROC (AUC = {mean_auc:.4f} \u00b1 {std_auc:.4f})")
    ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color=c_roc, alpha=0.2,
                     label="\u00b1 1 std. dev.")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.5, linestyle=':', label="Chance")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_title(f"ProGRATE test set ROC curve", fontweight="bold", fontsize=16, pad=15)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=False, prop={'weight':'bold', 'size':11})
    
    _style_axes(ax)
    _save_fig(fig, out_dir, f"roc_mean{tag}")

    return {"mean_auc": mean_auc, "std_auc": std_auc}

if __name__ == "__main__":
    from config import LOG_DIR, FIGURES_DIR
    
    fold_histories = os.path.join(str(LOG_DIR)), "cv_epoch_metrics.csv"
    fig_dir = str(FIGURES_DIR)
    plot_accuracy_curves(fold_histories, fig_dir, )