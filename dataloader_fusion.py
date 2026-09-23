import os
import math
import numpy as np
import torch
from collections import Counter
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset

from config import DATA_CONFIG
from utils import set_global_seed, print_class_distribution


def onehot_to_int(label_str: str) -> int:
    label_str = label_str.strip()
    if label_str == "10":
        return 0
    elif label_str == "01":
        return 1
    raise ValueError(f"Unexpected one-hot label: {label_str!r} (expected '10' or '01')")


def int_to_onehot(label_int: int) -> str:
    return "10" if label_int == 0 else "01"


def _is_numeric(token: str) -> bool:
    """Return True if token looks like a number."""
    try:
        float(token)
        return True
    except ValueError:
        return False

def parse_line(line: str) -> Tuple[str, List[str], str]:
    parts = line.strip().rstrip(".").split()
    parts = [p.rstrip(".") for p in parts]

    if len(parts) >= 4 and _is_numeric(parts[-1]) and parts[-2].upper().startswith("Q"):
        parts = parts[:-1]

    if len(parts) < 3:
        raise ValueError(f"Line too short to parse: {line!r}")

    label_str = parts[0]
    qprofile = parts[-1]
    motifs = parts[1:-1]

    if len(motifs) == 0:
        raise ValueError(f"No motifs found in line: {line!r}")

    return label_str, motifs, qprofile


def load_fusion_tsv(path: str) -> List[Dict]:
    """Load a .tsv file into a list of records: {label, motifs, qprofile}."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                label_str, motifs, qprofile = parse_line(line)
                label_int = onehot_to_int(label_str)
            except Exception as e:
                print(f"  [WARN] Skipping malformed line {line_no}: {e}")
                continue
            records.append({
                "label": label_int,
                "label_onehot": label_str,
                "motifs": motifs,
                "qprofile": qprofile,
                "mgm_sequence": " ".join(motifs),
                "sigd_tokens": motifs + [qprofile],
            })
    print(f"  Loaded {len(records)} valid records from: {path}")
    return records


def load_fusion_data(paths: List[str]) -> List[Dict]:
    """Load and concatenate one or more .tsv files."""
    all_records = []
    for p in paths:
        all_records.extend(load_fusion_tsv(p))
    labels = np.array([r["label"] for r in all_records])
    print_class_distribution(labels, name="Full fusion dataset")
    return all_records

def build_shared_split(records: List[Dict], test_size: float = None,
                        val_size: float = None, seed: int = 42) -> Dict[str, List[int]]:
    test_size = test_size if test_size is not None else DATA_CONFIG.test_size
    val_size = val_size if val_size is not None else DATA_CONFIG.val_size

    labels = np.array([r["label"] for r in records])
    idx = np.arange(len(records))

    idx_temp, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels, shuffle=True
    )
    idx_train, idx_val = train_test_split(
        idx_temp, test_size=val_size, random_state=seed,
        stratify=labels[idx_temp], shuffle=True
    )

    print(f"\nShared split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    print_class_distribution(labels[idx_train], name="Train")
    print_class_distribution(labels[idx_val], name="Val")
    print_class_distribution(labels[idx_test], name="Test")

    return {"train": idx_train.tolist(), "val": idx_val.tolist(), "test": idx_test.tolist()}


def build_vocab(train_records: List[Dict], min_freq: int = 1) -> Dict[str, int]:
    """Build word_id_map from TRAIN split only. id 0 reserved for <PAD/UNK>."""
    counter = Counter()
    for r in train_records:
        counter.update(r["sigd_tokens"])
    vocab = [w for w, c in counter.items() if c >= min_freq]
    word_id_map = {w: i + 1 for i, w in enumerate(vocab)}  # 0 = PAD/UNK
    print(f"  Vocab size (train, min_freq={min_freq}): {len(word_id_map)}")
    return word_id_map


def compute_pmi_adjacency(train_records: List[Dict], word_id_map: Dict[str, int]) -> np.ndarray:
    """Word-word PMI adjacency, treating each (short) sample as one window.

    Fixed for the whole training run — reused unchanged every batch.
    """
    V = len(word_id_map)
    word_occ = np.zeros(V, dtype=np.int64)
    pair_occ: Dict[Tuple[int, int], int] = {}
    total_docs = 0

    for r in train_records:
        ids = sorted(set(word_id_map[t] for t in r["sigd_tokens"] if t in word_id_map))
        if not ids:
            continue
        total_docs += 1
        for i in ids:
            word_occ[i - 1] += 1
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                key = (ids[a], ids[b])
                pair_occ[key] = pair_occ.get(key, 0) + 1

    pmi_adj = np.zeros((V, V), dtype=np.float32)
    for (i, j), count in pair_occ.items():
        wi, wj = word_occ[i - 1], word_occ[j - 1]
        if wi == 0 or wj == 0:
            continue
        pmi = math.log((count * max(total_docs, 1)) / (wi * wj))
        if pmi > 0:
            pmi_adj[i - 1, j - 1] = pmi
            pmi_adj[j - 1, i - 1] = pmi

    print(f"  PMI adjacency built: {V}x{V}, nonzero={np.count_nonzero(pmi_adj)}")
    return pmi_adj


def compute_word_doc_freq(train_records: List[Dict], word_id_map: Dict[str, int]) -> np.ndarray:
    """Document frequency per vocab word (for IDF), from TRAIN split only."""
    V = len(word_id_map)
    doc_freq = np.zeros(V, dtype=np.int64)
    for r in train_records:
        seen = set(word_id_map[t] for t in r["sigd_tokens"] if t in word_id_map)
        for i in seen:
            doc_freq[i - 1] += 1
    return doc_freq


def compute_idf_vector(train_records: List[Dict], word_id_map: Dict[str, int]) -> np.ndarray:
    N = len(train_records)
    doc_freq = compute_word_doc_freq(train_records, word_id_map)
    idf = np.log(N / (doc_freq + 1.0)).astype(np.float32)
    return idf


def compute_word_label_pr(train_records: List[Dict], word_id_map: Dict[str, int], num_classes: int = 2) -> np.ndarray:
    V = len(word_id_map)
    tf_by_class = np.zeros((num_classes, V), dtype=np.float64)

    for r in train_records:
        c = r["label"]
        for t in r["sigd_tokens"]:
            if t in word_id_map:
                tf_by_class[c, word_id_map[t] - 1] += 1.0

    pr_matrix = np.zeros((num_classes, V), dtype=np.float64)
    for c in range(num_classes):
        pos = tf_by_class[c]
        neg = tf_by_class.sum(axis=0) - pos
        pr_matrix[c] = (pos + 1e-10) / (neg + 1e-10)

    word_label_weight = (pr_matrix * tf_by_class).T.astype(np.float32) 

    if word_label_weight.max() > 0:
        word_label_weight = word_label_weight / (word_label_weight.max() + 1e-8)

    print(f"  Word-label PR weights built: {word_label_weight.shape}")
    return word_label_weight


def compute_graph_stats(train_records: List[Dict], word_id_map: Dict[str, int], num_classes: int = 2) -> Dict[str, np.ndarray]:
    print("\nComputing global graph statistics (train split only)...")
    pmi_adj = compute_pmi_adjacency(train_records, word_id_map)
    idf_vec = compute_idf_vector(train_records, word_id_map)
    word_label_weight = compute_word_label_pr(train_records, word_id_map, num_classes)
    return {
        "pmi_adj": pmi_adj,
        "idf_vec": idf_vec,
        "word_label_weight": word_label_weight,
    }

# ============================================================
def build_motif_embedding_cache(records: List[Dict], cache_path: str, force_recompute: bool = False) -> np.ndarray:
    from embeddings import extract_per_motif_embeddings

    sequences = [r["mgm_sequence"] for r in records]
    per_motif_emb = extract_per_motif_embeddings(
        sequences, cache_path, batch_size=256, force_recompute=force_recompute,
    )
    return per_motif_emb

class FusionDataset(Dataset):
    def __init__(self, records: List[Dict], indices: List[int],
                 motif_emb_full: np.ndarray, word_id_map: Dict[str, int]):
        self.records = [records[i] for i in indices]
        self.word_id_map = word_id_map

        self.motif_emb = motif_emb_full[indices]

        n, K, _ = self.motif_emb.shape
        self.motif_mask = np.empty((n, K), dtype=bool)


        chunk_size = 2048


        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = self.motif_emb[start:end]
            emb_norm = np.linalg.norm(chunk, axis=-1)
            self.motif_mask[start:end] = (emb_norm < 1e-6)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        motif_emb = torch.from_numpy(self.motif_emb[idx].astype(np.float32))
        motif_mask = torch.from_numpy(self.motif_mask[idx])
        sigd_ids = torch.tensor(
            [self.word_id_map.get(t, 0) for t in r["sigd_tokens"]], dtype=torch.long
        )
        label = torch.tensor(r["label"], dtype=torch.long)
        return motif_emb, motif_mask, sigd_ids, label


def fusion_collate_fn(batch, vocab_size: int):
    motif_embs, motif_masks, sigd_ids_list, labels = zip(*batch)

    motif_emb = torch.stack(motif_embs, dim=0)  
    motif_mask = torch.stack(motif_masks, dim=0) 
    labels = torch.stack(labels, dim=0)      

    max_len = max(len(ids) for ids in sigd_ids_list)
    B = len(sigd_ids_list)
    lstm_input_ids = torch.zeros(B, max_len, dtype=torch.long)
    lstm_pad_mask = torch.ones(B, max_len, dtype=torch.bool) 
    doc_word_freq = torch.zeros(B, vocab_size, dtype=torch.float32)

    for i, ids in enumerate(sigd_ids_list):
        L = len(ids)
        lstm_input_ids[i, :L] = ids
        lstm_pad_mask[i, :L] = False
        for tid in ids.tolist():
            if tid > 0: 
                doc_word_freq[i, tid - 1] += 1.0

    return motif_emb, motif_mask, lstm_input_ids, lstm_pad_mask, doc_word_freq, labels


def prepare_cv_folds(tsv_paths: List[str], cache_dir: str, n_splits: int = 10, seed: int = 42, force_recompute_embeddings: bool = False) -> Dict:
    set_global_seed(seed)

    records = load_fusion_data(tsv_paths)
    labels = np.array([r["label"] for r in records])

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "fusion_per_motif_embeddings.npy")
    print("\nBuilding / loading DNABERT per-motif embedding cache for ALL records...")
    motif_emb_full = build_motif_embedding_cache(
        records, cache_path, force_recompute=force_recompute_embeddings
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_indices = [
        (trainval_idx, test_idx)
        for trainval_idx, test_idx in skf.split(np.zeros(len(labels)), labels)
    ]

    print(f"\nPrepared {n_splits}-fold stratified CV splits "
          f"({len(records)} total records).")

    return {
        "records": records,
        "labels": labels,
        "motif_emb_full": motif_emb_full,
        "fold_indices": fold_indices,
    }


def build_fold_data(records: List[Dict], labels: np.ndarray, motif_emb_full: np.ndarray,
                     trainval_idx: np.ndarray, test_idx: np.ndarray,
                     val_size: float = None, num_classes: int = 2,
                     min_freq: int = 1, seed: int = 42, fold_num: int = None) -> Dict:

    val_size = val_size if val_size is not None else DATA_CONFIG.val_size
    fold_tag = f"[fold {fold_num}] " if fold_num is not None else ""

    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_size, random_state=seed,
        stratify=labels[trainval_idx], shuffle=True,
    )

    print(f"\n{fold_tag}split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print_class_distribution(labels[train_idx], name=f"{fold_tag}Train")
    print_class_distribution(labels[val_idx], name=f"{fold_tag}Val")
    print_class_distribution(labels[test_idx], name=f"{fold_tag}Test")

    train_records = [records[i] for i in train_idx]

    print(f"{fold_tag}Building vocabulary (train split only)...")
    word_id_map = build_vocab(train_records, min_freq=min_freq)
    graph_stats = compute_graph_stats(train_records, word_id_map, num_classes=num_classes)

    datasets = {
        "train": FusionDataset(records, train_idx, motif_emb_full, word_id_map),
        "val": FusionDataset(records, val_idx, motif_emb_full, word_id_map),
        "test": FusionDataset(records, test_idx, motif_emb_full, word_id_map),
    }

    return {
        "datasets": datasets,
        "word_id_map": word_id_map,
        "graph_stats": graph_stats,
        "split": {"train": train_idx, "val": val_idx, "test": test_idx},
    }


def load_fusion_pipeline(tsv_paths: List[str], cache_dir: str,
                          force_recompute_embeddings: bool = False) -> Dict:
    
    set_global_seed(42)

    records = load_fusion_data(tsv_paths)
    split = build_shared_split(records)

    train_records = [records[i] for i in split["train"]]

    print("\nBuilding vocabulary (train split only)...")
    word_id_map = build_vocab(train_records, min_freq=1)

    graph_stats = compute_graph_stats(train_records, word_id_map, num_classes=2)

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "fusion_per_motif_embeddings.npy")
    print("\nBuilding / loading DNABERT per-motif embedding cache for ALL records...")
    motif_emb_full = build_motif_embedding_cache(
        records, cache_path, force_recompute=force_recompute_embeddings
    )

    datasets = {
        split_name: FusionDataset(records, idx_list, motif_emb_full, word_id_map)
        for split_name, idx_list in split.items()
    }

    return {
        "datasets": datasets,
        "word_id_map": word_id_map,
        "graph_stats": graph_stats,
        "split": split,
        "records": records,
    }
