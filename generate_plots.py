import pandas as pd
from cv_report import plot_loss_curves, plot_accuracy_curves

CSV_PATH = "results/logs/cv_epoch_metrics.csv"
OUT_DIR = "results/figures"

print("Loading CSV data...")
df = pd.read_csv(CSV_PATH)

fold_histories = []
for fold_num, group in df.groupby("fold"):
    group = group.sort_values("epoch")
    fold_histories.append({
        "train_loss": group["train_loss"].tolist(),
        "train_acc": group["train_acc"].tolist(),
        "val_loss": group["val_loss"].tolist(),
        "val_acc": group["val_acc"].tolist(),
        "val_auc": group["val_auc"].tolist()
    })

print("Generating Loss Curve...")
plot_loss_curves(fold_histories, out_dir=OUT_DIR, tag=TAG)

print("Generating Accuracy Curve...")
plot_accuracy_curves(fold_histories, out_dir=OUT_DIR, tag=TAG)

print("✅ Done! Check your figures folder.")