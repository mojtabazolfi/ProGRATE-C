from pathlib import Path
from dataclasses import dataclass


PROJECT_ROOT = Path("")
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
LOG_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
REPORT_DIR = RESULTS_DIR / "reports"

for _d in [DATA_DIR, EMBEDDINGS_DIR, RESULTS_DIR, CHECKPOINT_DIR, LOG_DIR, FIGURES_DIR, PREDICTIONS_DIR, REPORT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

@dataclass
class DataConfig:
    test_size: float = 0.20
    val_size: float = 0.125 

@dataclass
class EmbeddingConfig:
    model_name: str = "Taykhoom/DNABERT-4mer"
    embedding_dim: int = 768

@dataclass
class MLETFormerConfig:
    motif_embedding_dim: int = 768     
    max_motif_count: int = 64       

    d_model: int = 256    
    ffn_dim: int = 512
    num_heads: int = 4
    dropout: float = 0.1

    seq_num_layers: int = 3
    seq_use_cls_token: bool = True

    set_num_sab_layers: int = 2
    set_num_seeds: int = 1

    use_cross_attention: bool = True
    fusion_num_heads: int = 4

    gate_temp_start: float = 5.0    
    gate_temp_end: float = 1.0  
    gate_anneal_epochs: int = 30    

@dataclass
class TrainConfig:
    lr_scheduler: str = "cosine_warmup" 
    max_epochs: int = 60      
    warmup_epochs: int = 5        
    min_lr: float = 1e-6          

DATA_CONFIG = DataConfig()
EMBED_CONFIG = EmbeddingConfig()
MLET_CONFIG = MLETFormerConfig()
TRAIN_CONFIG = TrainConfig()

def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()


if __name__ == "__main__":
    print("=" * 50)
    print("Fusion pipeline configuration")
    print("=" * 50)
    print(f"  Device:        {DEVICE}")
    print(f"  Seed:          {SEED}")
    print(f"  DNABERT model: {EMBED_CONFIG.model_name}")
    print(f"  d_model (MLET): {MLET_CONFIG.d_model}")
    print(f"  Split:         test={DATA_CONFIG.test_size}, val={DATA_CONFIG.val_size}")
    print(f"  Embeddings ->  {EMBEDDINGS_DIR}")
    print(f"  Checkpoints -> {CHECKPOINT_DIR}")
    print(f"  Logs ->        {LOG_DIR}")
    print("=" * 50)
