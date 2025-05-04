# latent.py  (NEW FILE)
import torch, math
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

    def _build(self, S:int, device):
        total = S * self.hidden_size
        L_new = _find_closest_divisor(total, self.target_L or max(2, S//2))
        self.dk     = self.hidden_size // self.nh_stack
        self.L_new  = L_new
        self.C_new  = total // L_new

        self.to_latent = torch.nn.Linear(self.C_new, self.d_new, bias=self.bias).to(device)
        self.seq_len   = S
        self._built = True

    def forward(self, x:torch.Tensor, mask:Optional[torch.Tensor]=None):
        B,S,H = x.shape
        if not self._built:              self._build(S, x.device)
        elif S != self.seq_len:          raise RuntimeError("varying seq len not supported (build once)")

        if mask is not None:             x = x * mask.unsqueeze(-1)  # zero PAD

        # head‑stack
        x_stacked = torch.cat(torch.split(x, self.dk, dim=-1), dim=1)   # [B,S*nh_stack,dk]
        # chunk
        z = x_stacked.view(B,-1).view(B,self.L_new,self.C_new)
        z = self.to_latent(z)                                            # [B,L_new,d_new]

        # derive boolean latent_pad_mask
        latent_pad = None
        if mask is not None:
            latent_pad = (z.abs().sum(-1) < 1e-9)   # True = PAD

        return z, latent_pad

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