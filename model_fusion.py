import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from config import MLET_CONFIG


class SequenceTransformerBranch(nn.Module):
    def __init__(self,
                 input_dim: int = None,
                 d_model: int = None,
                 num_heads: int = None,
                 num_layers: int = None,
                 ffn_dim: int = None,
                 dropout: float = None,
                 max_seq_len: int = None,
                 use_cls_token: bool = None):
        super().__init__()

        input_dim = input_dim or MLET_CONFIG.motif_embedding_dim
        d_model = d_model or MLET_CONFIG.d_model
        num_heads = num_heads or MLET_CONFIG.num_heads
        num_layers = num_layers or MLET_CONFIG.seq_num_layers
        ffn_dim = ffn_dim or MLET_CONFIG.ffn_dim
        dropout = dropout if dropout is not None else MLET_CONFIG.dropout
        max_seq_len = max_seq_len or MLET_CONFIG.max_motif_count
        use_cls_token = use_cls_token if use_cls_token is not None else MLET_CONFIG.seq_use_cls_token

        self.d_model = d_model
        self.use_cls_token = use_cls_token

        self.input_proj = nn.Linear(input_dim, d_model)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        pe_len = max_seq_len + (1 if use_cls_token else 0)
        self.pos_encoding = nn.Parameter(torch.zeros(1, pe_len, d_model))
        nn.init.normal_(self.pos_encoding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.size(0)
        x = self.input_proj(x)  # (B, K, d_model)

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)  # (B, K+1, d_model)
            if mask is not None:
                cls_mask = torch.zeros(B, 1, dtype=mask.dtype, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)  # (B, K+1)

        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x, src_key_padding_mask=mask)

        if self.use_cls_token:
            out = x[:, 0, :]  # CLS token
        else:
            if mask is not None:
                keep = (~mask).float().unsqueeze(-1)
                out = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1e-6)
            else:
                out = x.mean(dim=1)

        return self.output_norm(out)  # (B, d_model)


class MAB(nn.Module):

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int = None,
                 dropout: float = 0.1):
        super().__init__()
        ffn_dim = ffn_dim or d_model * 4
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, X: torch.Tensor, Y: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attention(X, Y, Y, key_padding_mask=key_padding_mask)
        H = self.norm1(X + attn_out)
        return self.norm2(H + self.ffn(H))


class SAB(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int = None,
                 dropout: float = 0.1):
        super().__init__()
        self.mab = MAB(d_model, num_heads, ffn_dim, dropout)

    def forward(self, X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.mab(X, X, key_padding_mask=mask)


class PMA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_seeds: int = 1,
                 ffn_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.S = nn.Parameter(torch.zeros(1, num_seeds, d_model))
        nn.init.normal_(self.S, std=0.02)
        self.rff = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.mab = MAB(d_model, num_heads, ffn_dim, dropout)

    def forward(self, X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = X.size(0)
        S = self.S.expand(B, -1, -1)
        return self.mab(S, self.rff(X), key_padding_mask=mask)  # (B, num_seeds, d_model)


class SetTransformerBranch(nn.Module):

    def __init__(self,
                 input_dim: int = None,
                 d_model: int = None,
                 num_heads: int = None,
                 num_sab_layers: int = None,
                 num_seeds: int = None,
                 ffn_dim: int = None,
                 dropout: float = None):
        super().__init__()

        input_dim = input_dim or MLET_CONFIG.motif_embedding_dim
        d_model = d_model or MLET_CONFIG.d_model
        num_heads = num_heads or MLET_CONFIG.num_heads
        num_sab_layers = num_sab_layers or MLET_CONFIG.set_num_sab_layers
        num_seeds = num_seeds or MLET_CONFIG.set_num_seeds
        ffn_dim = ffn_dim or MLET_CONFIG.ffn_dim
        dropout = dropout if dropout is not None else MLET_CONFIG.dropout

        self.input_proj = nn.Linear(input_dim, d_model)
        self.sab_layers = nn.ModuleList([
            SAB(d_model, num_heads, ffn_dim, dropout) for _ in range(num_sab_layers)
        ])
        self.pma = PMA(d_model, num_heads, num_seeds, ffn_dim, dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        for sab in self.sab_layers:
            x = sab(x, mask)
        x = self.pma(x, mask)     # (B, num_seeds, d_model)
        x = x.squeeze(1)          # assumes num_seeds == 1
        return self.output_norm(x)


class AdaptiveGating(nn.Module):

    def __init__(self,
                 d_model: int = None,
                 num_branches: int = 2,
                 temp_start: float = 5.0,
                 temp_end: float = 1.0,
                 anneal_epochs: int = 30):
        super().__init__()
        d_model = d_model or MLET_CONFIG.d_model
        self.num_branches = num_branches
        self.d_model = d_model

        self.register_buffer("temperature", torch.tensor(temp_start))
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.anneal_epochs = anneal_epochs

        self.gate_scorer = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate_scorer.weight)
        nn.init.zeros_(self.gate_scorer.bias)

    def set_temperature(self, epoch: int):
        if epoch >= self.anneal_epochs:
            t = self.temp_end
        else:
            progress = epoch / max(self.anneal_epochs, 1)
            t = self.temp_start + (self.temp_end - self.temp_start) * progress
        self.temperature.fill_(t)

    def forward(self, *branch_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(branch_outputs, dim=1)          # (B, num_branches, d_model)
        scores = self.gate_scorer(stacked).squeeze(-1)         # (B, num_branches)
        weights = F.softmax(scores / self.temperature, dim=1)  # (B, num_branches)
        gated = (stacked * weights.unsqueeze(-1)).sum(dim=1)   # (B, d_model)
        return gated, weights


class CrossAttentionFusion(nn.Module):
    def __init__(self,
                 d_model: int = None,
                 num_heads: int = None,
                 num_branches: int = 2,
                 dropout: float = None,
                 use_cross_attention: bool = None):
        super().__init__()

        d_model = d_model or MLET_CONFIG.d_model
        num_heads = num_heads or MLET_CONFIG.fusion_num_heads
        dropout = dropout if dropout is not None else MLET_CONFIG.dropout
        use_cross_attention = use_cross_attention if use_cross_attention is not None else MLET_CONFIG.use_cross_attention

        self.use_cross_attention = use_cross_attention
        self.num_branches = num_branches

        if use_cross_attention:
            self.q_proj = nn.Linear(d_model, d_model)
            self.kv_proj = nn.Linear(d_model, d_model)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True,
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.concat_proj = nn.Sequential(
                nn.Linear(d_model * num_branches, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )

    def forward(self, gated: torch.Tensor,
                branch_outputs: List[torch.Tensor]) -> torch.Tensor:
        if self.use_cross_attention:
            q = self.q_proj(gated).unsqueeze(1)          # (B, 1, d_model)
            kv = torch.stack(branch_outputs, dim=1)       # (B, num_branches, d_model)
            kv = self.kv_proj(kv)
            attn_out, _ = self.cross_attn(q, kv, kv)
            x = self.norm1(q + attn_out)
            x = self.norm2(x + self.ffn(x))
            return x.squeeze(1)
        concat = torch.cat(branch_outputs, dim=1)         # (B, num_branches * d_model)
        return self.concat_proj(concat)

class LSTMEncoderBranch(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 300,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        # +1 for PAD/UNK index 0
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=embed_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = embed_dim * 2  # bidirectional concat

    def forward(self, input_ids: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)   # (B, L, E)
        out, _ = self.lstm(x)            # (B, L, 2E)

        # Self-attention pooling (masked)
        d_k = out.size(-1)
        scores = torch.matmul(out, out.transpose(1, 2)) / (d_k ** 0.5)  # (B, L, L)
        key_mask = pad_mask.unsqueeze(1)  # (B, 1, L)
        scores = scores.masked_fill(key_mask, float("-inf"))
        alpha = F.softmax(scores, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)  # rows that are all-padding

        context = torch.matmul(alpha, out)  # (B, L, 2E)

        # Mean-pool over non-padded query positions
        keep = (~pad_mask).float().unsqueeze(-1)  # (B, L, 1)
        pooled = (context * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1e-6)
        return pooled  # (B, out_dim)


class FeaturelessGraphConv(nn.Module):

    def __init__(self, num_word_label_nodes: int, out_features: int, resrate: float = 0.1):
        super().__init__()
        self.out_features = out_features
        self.resrate = resrate
        self.weight = nn.Parameter(torch.empty(num_word_label_nodes, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(1, out_features))

    def forward(self, doc_features: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        support = torch.cat([doc_features, self.weight], dim=0)  # (B+V+C, out_features)
        out = torch.matmul(adj_norm, support) + self.bias
        return out + self.resrate * support

class GCNBranch(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, out_features: int,
                 pmi_adj: torch.Tensor, idf_vec: torch.Tensor,
                 word_label_weight: torch.Tensor):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.out_features = out_features

        self.gc1 = FeaturelessGraphConv(vocab_size + num_classes, out_features)

        self.register_buffer("pmi_adj", pmi_adj.float())                      # (V, V)
        self.register_buffer("idf_vec", idf_vec.float())                      # (V,)
        self.register_buffer("word_label_weight", word_label_weight.float())  # (V, C)

    def _build_batch_adjacency(self, B: int, doc_word_freq: torch.Tensor) -> torch.Tensor:
        V, C = self.vocab_size, self.num_classes
        N = B + V + C
        device = doc_word_freq.device

        adj = torch.zeros(N, N, device=device)

        # doc-word block (weighted by IDF * freq)
        doc_word_w = doc_word_freq * self.idf_vec.unsqueeze(0)  # (B, V)
        adj[:B, B:B + V] = doc_word_w
        adj[B:B + V, :B] = doc_word_w.T

        # word-word block (fixed PMI)
        adj[B:B + V, B:B + V] = self.pmi_adj

        # word-label block (fixed PR)
        adj[B:B + V, B + V:] = self.word_label_weight
        adj[B + V:, B:B + V] = self.word_label_weight.T

        return adj

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        N = adj.size(0)
        adj = adj + torch.eye(N, device=adj.device)  # self-loops
        rowsum = adj.sum(dim=1).clamp(min=1e-8)
        d_inv_sqrt = rowsum.pow(-0.5)
        d_mat = torch.diag(d_inv_sqrt)
        return d_mat @ adj @ d_mat

    def forward(self, doc_features: torch.Tensor, doc_word_freq: torch.Tensor) -> torch.Tensor:
        B = doc_features.size(0)
        adj = self._build_batch_adjacency(B, doc_word_freq)
        adj_norm = self._normalize_adj(adj)
        out = self.gc1(doc_features, adj_norm)  # (B+V+C, out_features)
        out = F.relu(out)
        return out[:B]  # doc rows only

class FusionModel(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 pmi_adj: torch.Tensor,
                 idf_vec: torch.Tensor,
                 word_label_weight: torch.Tensor,
                 num_classes: int = 2,
                 lstm_embed_dim: int = 300,
                 fusion_d_model: int = 128,
                 dropout: float = 0.1):
        super().__init__()

        motif_dim = MLET_CONFIG.motif_embedding_dim
        MLET_d_model = MLET_CONFIG.d_model 

        self.branch_seq = SequenceTransformerBranch(input_dim=motif_dim, d_model=MLET_d_model, dropout=dropout)
        self.branch_set = SetTransformerBranch(input_dim=motif_dim, d_model=MLET_d_model, dropout=dropout)
        self.MLET_gate = AdaptiveGating(
            d_model=MLET_d_model, num_branches=2,
            temp_start=MLET_CONFIG.gate_temp_start, temp_end=MLET_CONFIG.gate_temp_end,
            anneal_epochs=MLET_CONFIG.gate_anneal_epochs,
        )
        self.MLET_fusion = CrossAttentionFusion(d_model=MLET_d_model, num_branches=2)
        self.MLET_head = nn.Sequential(
            nn.LayerNorm(MLET_d_model),
            nn.Linear(MLET_d_model, fusion_d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        ) 

        self.lstm_encoder = LSTMEncoderBranch(vocab_size, embed_dim=lstm_embed_dim, dropout=dropout)
        self.gcn_branch = GCNBranch(
            vocab_size=vocab_size, num_classes=num_classes,
            out_features=self.lstm_encoder.out_dim,
            pmi_adj=pmi_adj, idf_vec=idf_vec, word_label_weight=word_label_weight,
        )
        self.gcn_proj = nn.Sequential(
            nn.Linear(self.lstm_encoder.out_dim, fusion_d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )  

        self.final_gate = AdaptiveGating(d_model=fusion_d_model, num_branches=2,
                                         temp_start=3.0, temp_end=1.0, anneal_epochs=20)
        self.final_fusion = CrossAttentionFusion(d_model=fusion_d_model, num_branches=2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_d_model),
            nn.Linear(fusion_d_model, fusion_d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_d_model // 2, num_classes),
        )

    def set_epoch(self, epoch: int):
        self.MLET_gate.set_temperature(epoch)
        self.final_gate.set_temperature(epoch)

    def forward(self,
                motif_emb: torch.Tensor, motif_mask: torch.Tensor,
                lstm_input_ids: torch.Tensor, lstm_pad_mask: torch.Tensor,
                doc_word_freq: torch.Tensor,
                return_gate_weights: bool = False
                ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:

        seq_out = self.branch_seq(motif_emb, motif_mask)
        set_out = self.branch_set(motif_emb, motif_mask)
        MLET_gated, MLET_gate_w = self.MLET_gate(seq_out, set_out)
        MLET_fused = self.MLET_fusion(MLET_gated, [seq_out, set_out])
        MLET_repr = self.MLET_head(MLET_fused)

        doc_feat = self.lstm_encoder(lstm_input_ids, lstm_pad_mask) 
        gcn_out = self.gcn_branch(doc_feat, doc_word_freq)  
        gcn_repr = self.gcn_proj(gcn_out) 

        final_gated, final_gate_w = self.final_gate(MLET_repr, gcn_repr)
        final_fused = self.final_fusion(final_gated, [MLET_repr, gcn_repr])
        logits = self.classifier(final_fused)

        gate_weights = None
        if return_gate_weights:
            gate_weights = {"MLET_internal": MLET_gate_w, "final": final_gate_w}

        return logits, gate_weights


def build_fusion_model(vocab_size: int, graph_stats: Dict[str, np.ndarray],
                       num_classes: int = 2, device: str = "cuda") -> FusionModel:
    pmi_adj = torch.from_numpy(np.asarray(graph_stats["pmi_adj"]))
    idf_vec = torch.from_numpy(np.asarray(graph_stats["idf_vec"]))
    word_label_weight = torch.from_numpy(np.asarray(graph_stats["word_label_weight"]))

    model = FusionModel(
        vocab_size=vocab_size,
        pmi_adj=pmi_adj, idf_vec=idf_vec, word_label_weight=word_label_weight,
        num_classes=num_classes,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nFusion Model built:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  MLET branch d_model: {MLET_CONFIG.d_model} -> 128")
    print(f"  LSTM out dim: {model.lstm_encoder.out_dim} -> GCN(gc1) -> 128")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    print(f"  Device: {device}")

    return model

if __name__ == "__main__":
    torch.manual_seed(0)
    B, K, V, C = 4, 16, 20, 2
    D = MLET_CONFIG.motif_embedding_dim

    graph_stats = {
        "pmi_adj": np.random.rand(V, V).astype(np.float32),
        "idf_vec": np.random.rand(V).astype(np.float32),
        "word_label_weight": np.random.rand(V, C).astype(np.float32),
    }
    model = build_fusion_model(V, graph_stats, num_classes=C, device="cpu")
    model.set_epoch(0)

    motif_emb = torch.randn(B, K, D)
    motif_mask = torch.zeros(B, K, dtype=torch.bool)
    motif_mask[0, 12:] = True  # some padding
    lstm_ids = torch.randint(0, V + 1, (B, 6))
    lstm_mask = torch.zeros(B, 6, dtype=torch.bool)
    doc_word_freq = torch.rand(B, V)
    labels = torch.randint(0, C, (B,))

    logits, gate_w = model(motif_emb, motif_mask, lstm_ids, lstm_mask, doc_word_freq,
                           return_gate_weights=True)
    print(f"\n  logits shape: {tuple(logits.shape)} (expected ({B}, {C}))")
    print(f"  gate keys: {list(gate_w.keys())}")

    from utils import FocalLoss
    loss = FocalLoss()(logits, labels)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  loss: {loss.item():.4f} | grad norm: {grad_norm:.4f}")
    print("\n[OK] FusionModel forward/backward test passed!")
