import os
import random
import numpy as np
import torch
from typing import Dict, Any, Optional
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, roc_auc_score,
    f1_score, matthews_corrcoef, average_precision_score,
)

from config import SEED


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:

    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)

    # Safety: FP16 overflow during AMP can produce NaN/Inf -> neutral 0.5
    if np.any(~np.isfinite(y_pred_proba)):
        n_bad = int(np.sum(~np.isfinite(y_pred_proba)))
        print(f"  [WARN] {n_bad} NaN/Inf values in y_pred_proba, replacing with 0.5")
        y_pred_proba = np.where(np.isfinite(y_pred_proba), y_pred_proba, 0.5)
    y_pred_proba = np.clip(y_pred_proba, 0.0, 1.0)

    y_pred = (y_pred_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    def safe_div(a, b):
        return float(a) / float(b) if b > 0 else 0.0

    metrics = {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": safe_div(tp, tp + fn),   # = recall for class 1
        "specificity": safe_div(tn, tn + fp),
        "precision": safe_div(tp, tp + fp),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
        "auprc": average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
    }

    for key in ["accuracy", "balanced_accuracy", "sensitivity", "specificity",
                "precision", "f1", "f1_macro", "mcc", "auc", "auprc"]:
        metrics[key] = round(metrics[key], 4)

    return metrics


def format_metrics_table(metrics: Dict[str, float]) -> str:
    lines = ["=" * 50, f"{'Metric':<25} {'Value':>10}", "-" * 50]
    display_keys = [
        "accuracy", "balanced_accuracy", "sensitivity", "specificity",
        "precision", "f1", "f1_macro", "mcc", "auc", "auprc",
    ]
    for key in display_keys:
        if key in metrics:
            lines.append(f"{key:<25} {metrics[key]:>10.4f}")
    lines.append("=" * 50)
    return "\n".join(lines)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2 and logits.size(1) > 1:
            return self._focal_loss_multiclass(logits, targets)
        return self._focal_loss_binary(logits.view(-1), targets.view(-1).float())

    def _focal_loss_multiclass(self, logits, targets):
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        if logits.size(1) == 2:
            alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        else:
            alpha_t = torch.ones_like(targets, dtype=torch.float)

        focal_loss = -alpha_t * (1 - pt).pow(self.gamma) * log_pt
        return self._reduce(focal_loss)

    def _focal_loss_binary(self, logits, targets):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t).pow(self.gamma) * bce_loss
        return self._reduce(focal_loss)

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def cosine_scheduler(optimizer, max_epochs: int, min_lr: float = 1e-6):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=min_lr,
    )


def cosine_warmup_scheduler(optimizer, max_epochs: int, warmup_epochs: int = 3,
                            min_lr: float = 1e-6):
    import math
    initial_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress))) + \
            (min_lr / initial_lr) * 0.5 * (1.0 - math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(optimizer, scheduler_name: str = None, max_epochs: int = None):
    from config import TRAIN_CONFIG
    scheduler_name = scheduler_name or TRAIN_CONFIG.lr_scheduler
    max_epochs = max_epochs or TRAIN_CONFIG.max_epochs

    if scheduler_name == "cosine_warmup":
        return cosine_warmup_scheduler(
            optimizer, max_epochs=max_epochs,
            warmup_epochs=TRAIN_CONFIG.warmup_epochs, min_lr=TRAIN_CONFIG.min_lr,
        )
    if scheduler_name == "cosine":
        return cosine_scheduler(optimizer, max_epochs=max_epochs, min_lr=TRAIN_CONFIG.min_lr)
    if scheduler_name in ("none", None):
        return None
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


class EarlyStopping:
    def __init__(self, patience: int = 12, min_delta: float = 0.0,
                 mode: str = "min", save_path: Optional[str] = None):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode 
        self.save_path = save_path

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score: float, model: torch.nn.Module,
                 epoch: int) -> bool:
        if self.best_score is None or self._is_improvement(current_score):
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta

    def _save_checkpoint(self, model: torch.nn.Module):
        if self.save_path:
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_score": self.best_score,
                "best_epoch": self.best_epoch,
            }, self.save_path)

    def load_best(self, model: torch.nn.Module):
        if self.save_path and os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Loaded best checkpoint from epoch {checkpoint['best_epoch']} "
                  f"with score {checkpoint['best_score']:.4f}")

def save_json(data: Dict[str, Any], path: str, indent: int = 2):
    """Save a dict to a JSON file (creating parent dirs as needed)."""
    import json
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

def get_class_distribution(y: np.ndarray, name: str = "") -> Dict[str, Any]:
    """Class-distribution stats from labels."""
    y = np.asarray(y).astype(int)
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {
        "name": name,
        "total": total,
        "classes": classes.tolist(),
        "counts": counts.tolist(),
        "proportions": [round(c / total, 4) for c in counts],
        "imbalance_ratio": round(max(counts) / min(counts), 4) if min(counts) > 0 else float("inf"),
    }


def print_class_distribution(y: np.ndarray, name: str = ""):
    """Pretty-print a class distribution."""
    dist = get_class_distribution(y, name)
    print(f"\nClass distribution [{dist['name']}]:")
    print(f"  Total samples: {dist['total']}")
    for cls, cnt, prop in zip(dist["classes"], dist["counts"], dist["proportions"]):
        label = "Control (0)" if cls == 0 else "Cancer (1)"
        print(f"  Class {cls} ({label}): {cnt} ({prop * 100:.2f}%)")
    print(f"  Imbalance ratio: {dist['imbalance_ratio']}")

if __name__ == "__main__":
    print("Testing utils.py ...")
    set_global_seed(42)

    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_proba = np.array([0.1, 0.3, 0.2, 0.8, 0.9, 0.7, 0.4, 0.6, 0.55, 0.95])
    print("\n" + format_metrics_table(compute_metrics(y_true, y_proba)))

    fl = FocalLoss(alpha=0.5, gamma=2.0)
    logits = torch.randn(8, 2, requires_grad=True)
    targets = torch.randint(0, 2, (8,))
    loss = fl(logits, targets)
    loss.backward()
    print(f"\n  Focal loss (multiclass): {loss.item():.4f}")

    print_class_distribution(y_true, name="test")
    print("\n[OK] utils tests passed!")
