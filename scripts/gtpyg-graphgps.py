import torch
from torch import nn

from gtpyg.gt_pyg.nn.gt_conv import GTConv

from scripts.graphsage import custom_data, custom_collate, custom_loss_func
from scripts.transformerconv import custom_create_graph
from scripts.graphtransformer import compute_laplacian_pe
        
class custom_gnn(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.pe = None
        self.k = 8
        self.gnn = GraphGPSRanker(
            in_dim=384,    # (384 + brand + color)
            edge_in_dim=1, # 2 (brand/color one-hot)
            hidden_dim=512,
            out_dim=384,
            pe_dim=self.k,
            num_layers=3,
            heads=4,
            dropout=0.1
        ).to(device)
       
    def forward(self, embeddings, edge_index, edge_attr):
        res = self.gnn(embeddings.to(self.device), edge_index.to(self.device), edge_attr.to(self.device), self.pe)
        if isinstance(res,tuple):
            return res[0]
        return res
        
    def get_node_emb(self, *kwargs):
        if self.pe == None:
            self.pe = compute_laplacian_pe(kwargs[0], self.k).to(self.device)
        return self.forward(kwargs[1], kwargs[2], kwargs[3])


class GPSBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_in_dim: int, heads: int = 4, attn_dropout: float = 0.1, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        # Local graph transformer (uses gt-pyg GTConv)
        self.local = GTConv(
            node_in_dim=hidden_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_heads=heads,
        )
        self.local_ln = nn.LayerNorm(hidden_dim)
        self.local_drop = nn.Dropout(dropout)

        # Global Transformer (vanilla PyTorch)
        self.global_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ff_mult * hidden_dim,
            dropout=attn_dropout,
            batch_first=True,  # (B, N, D)
            activation='gelu'
        )
        self.global_enc = nn.TransformerEncoder(self.global_layer, num_layers=1)
        self.global_ln = nn.LayerNorm(hidden_dim)
        self.global_drop = nn.Dropout(dropout)

        # MLP head inside the block
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_mult * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * hidden_dim, hidden_dim),
        )
        self.ff_ln = nn.LayerNorm(hidden_dim)
        self.ff_drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        # Local
        h_local = self.local(x, edge_index, edge_attr)   # [N, H] or tuple
        if isinstance(h_local, tuple):
            h_local = h_local[0]
        x = self.local_ln(x + self.local_drop(h_local))

        # Global (treat the whole graph as a single sequence)
        h_global = self.global_enc(x.unsqueeze(0))       # [1, N, H]
        h_global = h_global.squeeze(0)                   # [N, H]
        x = self.global_ln(x + self.global_drop(h_global))

        # Feed-forward
        h_ff = self.ff(x)
        x = self.ff_ln(x + self.ff_drop(h_ff))
        return x

# ---------- Full model ----------
class GraphGPSRanker(nn.Module):
    def __init__(self, in_dim: int, edge_in_dim: int, hidden_dim: int = 512, out_dim: int = 384,
                 pe_dim: int = 0, num_layers: int = 3, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.pe_proj = nn.Linear(pe_dim, hidden_dim) if pe_dim > 0 else None
        self.blocks = nn.ModuleList([
            GPSBlock(hidden_dim, edge_in_dim, heads=heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr, pe=None):
        h = self.in_proj(x)
        if self.pe_proj is not None and pe is not None:
            h = h + self.pe_proj(pe)
        for blk in self.blocks:
            h = blk(h, edge_index, edge_attr)
        return self.out_proj(h)  # [N, out_dim]


