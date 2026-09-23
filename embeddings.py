import os
import time
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Optional
from config import EMBED_CONFIG, DEVICE


class DNABertPerMotifEmbedder:
    def __init__(self, model_name: str = None, device: str = None):
        from transformers import BertTokenizer, BertModel

        self.model_name = model_name or EMBED_CONFIG.model_name
        self.device = device or DEVICE
        self.embedding_dim = EMBED_CONFIG.embedding_dim

        print(f"Loading DNABERT-4mer from: {self.model_name}")
        print(f"Device: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def embed_motifs_batch(self, motifs: List[str]) -> np.ndarray:
        if not motifs:
            return np.zeros((0, self.embedding_dim), dtype=np.float16)

        inputs = self.tokenizer(
            motifs, return_tensors="pt", padding=True, truncation=True,
            max_length=10,  # motifs are only 4 characters
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        token_reps = outputs.last_hidden_state  # (B, L, D)

        batch_lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(1)

        batch_size, _, dim = token_reps.shape
        embeddings = torch.zeros(batch_size, dim, device=token_reps.device)
        for i, tokens_len in enumerate(batch_lens):
            if tokens_len <= 2:
                embeddings[i] = token_reps[i, 0, :]
            else:
                embeddings[i] = token_reps[i, 1:tokens_len - 1].mean(dim=0)

        return embeddings.cpu().numpy().astype(np.float16)


def extract_per_motif_embeddings(
    sequences: List[str],
    output_path: str,
    batch_size: int = 256,
    force_recompute: bool = False,
    show_progress: bool = True,
) -> np.ndarray:

    if not force_recompute and os.path.exists(output_path):
        print(f"Loading cached per-motif embeddings from: {output_path}")
        embeddings = np.load(output_path)
        if embeddings.shape[0] == len(sequences):
            print(f"  Loaded {embeddings.shape[0]} samples, "
                  f"K={embeddings.shape[1]}, dim={embeddings.shape[2]}")
            print(f"  Memory: {embeddings.nbytes / (1024 ** 2):.1f} MB")
            return embeddings
        print(f"  Cache size mismatch ({embeddings.shape[0]} vs {len(sequences)}), re-extracting...")

    print("Flattening motifs from all samples...")
    all_motifs: List[str] = []
    sample_motif_counts: List[int] = []
    for seq in sequences:
        motifs = seq.split()
        sample_motif_counts.append(len(motifs))
        all_motifs.extend(motifs)

    total_motifs = len(all_motifs)
    K = max(sample_motif_counts) if sample_motif_counts else 0
    N = len(sequences)
    D = EMBED_CONFIG.embedding_dim

    print(f"  Total samples: {N}")
    print(f"  Total motifs: {total_motifs}")
    print(f"  Max motifs per sample (K): {K}")
    print(f"  Embedding dim: {D}")
    print(f"  Expected output size: {N * K * D * 2 / (1024 ** 2):.1f} MB (float16)")

    embedder = DNABertPerMotifEmbedder()

    n_batches = (total_motifs + batch_size - 1) // batch_size
    all_embeddings = np.zeros((total_motifs, D), dtype=np.float16)

    print(f"\nExtracting per-motif embeddings in {n_batches} batches "
          f"(batch_size={batch_size} motifs)...")
    start_time = time.time()

    iterator = range(0, total_motifs, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Per-motif embedding", unit="batch", total=n_batches)

    for start in iterator:
        end = min(start + batch_size, total_motifs)
        all_embeddings[start:end] = embedder.embed_motifs_batch(all_motifs[start:end])

        if show_progress and (start // batch_size) % 50 == 0 and hasattr(iterator, "set_postfix"):
            elapsed = time.time() - start_time
            rate = end / max(elapsed, 1e-6)
            iterator.set_postfix({
                "rate": f"{rate:.0f} motifs/s",
                "ETA": f"{(total_motifs - end) / max(rate, 1e-6) / 60:.1f} min",
            })

    total_time = time.time() - start_time
    print(f"\nExtraction complete in {total_time / 60:.2f} minutes "
          f"({total_motifs / max(total_time, 1):.0f} motifs/sec)")

    print("Reshaping to (N, K, D)...")
    per_motif_emb = np.zeros((N, K, D), dtype=np.float16)
    idx = 0
    for i, count in enumerate(sample_motif_counts):
        per_motif_emb[i, :count, :] = all_embeddings[idx:idx + count, :]
        idx += count

    print(f"  Output shape: {per_motif_emb.shape} ({per_motif_emb.nbytes / (1024 ** 2):.1f} MB)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, per_motif_emb)
    print(f"  Saved to: {output_path}")

    return per_motif_emb
