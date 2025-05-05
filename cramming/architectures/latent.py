# latent.py  (NEW FILE)
import torch, math
import warnings
from typing import Optional
from .attention import _LayerNorm, _LatentAttentionInternal, _LatentMLPInternal, _find_closest_divisor

class InitialLatentTransform(torch.nn.Module):
    """
    Single module that converts [B,S,H] → Z, latent_mask
    (exactly what LMABertAttention did up to `to_latent`)
    """
    def __init__(self, hidden_size:int, cfg_attn):
        super().__init__()
        self.hidden_size = hidden_size
        self.nh_stack = int(cfg_attn.num_heads_stacking)
        self.d_new    = int(cfg_attn.d_new)
        self.bias     = cfg_attn.qkv_bias
        self.dropout  = torch.nn.Dropout(cfg_attn.dropout_prob)

        self._built = False
        self.target_L = getattr(cfg_attn, "target_l_new", None)

        # ------------------------------------------------------------------
        #  Static build: required when torch.compile(fullgraph=True) because
        #  parameter creation is **not** allowed inside the forward pass.
        #  We therefore build the Linear projection once here, using the
        #  sequence length configured in the attention config.
        # ------------------------------------------------------------------
        static_S = getattr(cfg_attn, "static_seq_len", None)
        if static_S is None:
            raise ValueError(
                "InitialLatentTransform requires `cfg_attn.static_seq_len` "
                "when torch.compile(fullgraph=True); please set it in the "
                "YAML (e.g. 128 for BERT‑base)."
            )
        # Build parameters on CPU – they will be moved to the correct device
        # by the first `.to(device)` call in the main model.
        self._build(static_S, torch.device("cpu"))

    def _build(self, S:int, device):
        total = S * self.hidden_size
        L_new = _find_closest_divisor(total, self.target_L or max(2, S//2))
        self.dk     = self.hidden_size // self.nh_stack
        self.L_new  = L_new
        self.C_new  = total // L_new

        self.to_latent = torch.nn.Linear(self.C_new, self.d_new, bias=self.bias).to(device)
        # linear to return from latent space
        self.from_latent = torch.nn.Linear(self.d_new, self.C_new, bias=self.bias).to(device)
        self.seq_len   = S
        self._built = True

    def forward(self, x:torch.Tensor, mask:Optional[torch.Tensor]=None):
        B,S,H = x.shape
        if S != self.seq_len:
            raise RuntimeError(
                f"InitialLatentTransform was built for seq_len={self.seq_len}, "
                f"but received seq_len={S}.  Either pad/trim sequences to a "
                f"fixed length or rebuild the model with the desired length."
            )

        if mask is None:
            warnings.warn(
                "InitialLatentTransform received no mask: padding positions will not be zeroed. "
                "Please provide mask to avoid potential contamination.",
                UserWarning,
            )
        if mask is not None:             x = x * mask.unsqueeze(-1)  # zero PAD

        # head‑stack
        x_stacked = torch.cat(torch.split(x, self.dk, dim=-1), dim=1)   # [B,S*nh_stack,dk]
        # chunk
        z = x_stacked.view(B,-1).view(B,self.L_new,self.C_new)
        z = self.to_latent(z)                                            # [B,L_new,d_new]

        # derive boolean latent_pad_mask
        latent_pad = None
        if mask is not None:
            # ------------------------------------------------------------------
            # Derive robust latent‑space padding mask.
            #  • mask : [B,S,1]  (1 = real token, 0 = PAD)
            #  • A latent position is PAD only if *all* contributing S·H elements
            #    were PAD.  We therefore expand the token‑mask over the hidden
            #    dimension, follow the same flatten‑and‑chunk pattern as the
            #    data path, and then mark PAD when the chunk sum is zero.
            # ------------------------------------------------------------------
            B = mask.size(0)
            tok_mask = mask.squeeze(-1)                     # [B,S]   1/0
            mask_expanded_flat = tok_mask.unsqueeze(-1) \
                                        .repeat(1, 1, self.hidden_size) \
                                        .view(B, -1)        # [B,S*H]
            mask_chunks = mask_expanded_flat.view(B, self.L_new, self.C_new)
            latent_pad = (mask_chunks.sum(dim=-1) == 0)     # [B,L_new]  bool
            # Zero‑out latent vectors that originate solely from padding
            z = z.masked_fill(latent_pad.unsqueeze(-1), 0.0)

        return z, latent_pad

    # ------------------------------------------------------------------
    def inverse_transform(self, z_latent: torch.Tensor) -> torch.Tensor:
        """
        Map latent tensor [B, L_new, d_new] back to original hidden space
        [B, S, H].  No masking logic here – caller can zero PAD tokens.
        """
        B, L_new, _ = z_latent.shape
        # latent -> chunk features
        chunks_back = self.from_latent(z_latent)              # [B, L_new, C_new]
        flat_back   = chunks_back.reshape(B, -1)              # [B, S*H]
        x_stacked   = flat_back.view(B, self.seq_len * self.nh_stack, self.dk)
        # un‑stack heads
        x_unstack   = x_stacked.view(B, self.seq_len, self.nh_stack, self.dk)
        out         = torch.cat(torch.unbind(x_unstack, dim=2), dim=2)  # [B,S,H]
        return out

class LatentLayer(torch.nn.Module):
    """A *single* Transformer block that works *entirely* in latent space."""
    __constants__ = ["LAYOUT"]
    LAYOUT: str = "[B S H]"
    def __init__(self, d_new:int, nh_latent:int, ff_hidden:int, dropout:float, bias:bool):
        super().__init__()
        self.ln1   = _LayerNorm(d_new, bias)
        self.attn  = _LatentAttentionInternal(d_new, nh_latent, dropout, bias)
        self.drop1 = torch.nn.Dropout(dropout)

        self.ln2   = _LayerNorm(d_new, bias)
        self.mlp   = _LatentMLPInternal(d_new, ff_hidden, dropout, bias)
        self.drop2 = torch.nn.Dropout(dropout)

    def forward(self, z, pad_mask:Optional[torch.Tensor]=None):
        z = z + self.drop1(self.attn(self.ln1(z), pad_mask))
        z = z + self.drop2(self.mlp(self.ln2(z)))
        return z