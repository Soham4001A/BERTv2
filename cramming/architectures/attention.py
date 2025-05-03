"""Attention modules. The final model uses "self-attention", but other options were tried and are still documented here."""
import torch
from transformers.models.bert.modeling_bert import BertSelfAttention
from .embeddings import Rotary, RotarySanityCheck, RotaryEleutherAI, RotaryLLAMA
from typing import Optional
from einops.layers.torch import Rearrange
from einops import rearrange

# ────────────────────────────────────────────────────────────────────────────
# Latent Meta Attention – NLP variant
# Ports the RL implementation to '[B,S,H]' tensors used in BERT.
# ────────────────────────────────────────────────────────────────────────────
import types

def _find_closest_divisor(n: int, target: int) -> int:
    """
    Find a divisor of n that is as close as possible to 'target'.
    Guarantees the return value divides n.
    """
    if n % target == 0:
        return target
    # scan outwards from target
    for offset in range(1, n):
        lo = target - offset
        hi = target + offset
        if lo > 0 and n % lo == 0:
            return lo
        if hi <= n and n % hi == 0:
            return hi
    return 1  # fallback (n is divisible by 1)

class _LayerNorm(torch.nn.Module):
    """Simple LayerNorm (bias optional) — duplicated to avoid circular import."""
    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias   = torch.nn.Parameter(torch.zeros(dim)) if bias else None
    def forward(self, x):
        return torch.nn.functional.layer_norm(x, self.weight.shape,
                                              self.weight,
                                              self.bias, 1e-5)

# ---- Latent‑space primitives copied from the RL version -------------------
class _LatentAttention(torch.nn.Module):
    """Multi‑head attention operating in latent '[B,L_new,d_new]' space."""
    def __init__(self, d_new: int, num_heads: int, dropout: float, bias: bool):
        super().__init__()
        assert d_new % num_heads == 0, "d_new must be divisible by num_heads"
        self.d_new   = d_new
        self.nh      = num_heads
        self.dk      = d_new // num_heads
        self.flash   = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.c_attn  = torch.nn.Linear(d_new, 3 * d_new, bias=bias)
        self.c_proj  = torch.nn.Linear(d_new, d_new, bias=bias)
        self.at_drop = torch.nn.Dropout(dropout)
        self.res_drop= torch.nn.Dropout(dropout)

    def forward(self, z):
        B, L, _ = z.shape
        q, k, v = self.c_attn(z).chunk(3, dim=-1)                   # [B,L,d_new]×3
        q = q.view(B, L, self.nh, self.dk).transpose(1, 2)          # [B,nh,L,dk]
        k = k.view(B, L, self.nh, self.dk).transpose(1, 2)
        v = v.view(B, L, self.nh, self.dk).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.at_drop.p if self.training else 0.0,
                    is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / self.dk ** 0.5)
            att = torch.nn.functional.softmax(att, dim=-1)
            att = self.at_drop(att)
            y   = att @ v                                              # [B,nh,L,dk]
        y = y.transpose(1, 2).reshape(B, L, self.d_new)               # [B,L,d_new]
        return self.res_drop(self.c_proj(y))

class _LatentMLP(torch.nn.Module):
    """Feed‑forward network in latent space."""
    def __init__(self, d_new: int, hidden: int, dropout: float, bias: bool):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_new, hidden, bias=bias),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, d_new, bias=bias),
            torch.nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class _LMABlock(torch.nn.Module):
    """A single transformer block in latent space."""
    def __init__(self, d_new: int, num_heads: int, ff_hidden: int,
                 dropout: float, bias: bool):
        super().__init__()
        self.ln1  = _LayerNorm(d_new, bias)
        self.attn = _LatentAttention(d_new, num_heads, dropout, bias)
        self.ln2  = _LayerNorm(d_new, bias)
        self.mlp  = _LatentMLP(d_new, ff_hidden, dropout, bias)
    def forward(self, z):
        z = z + self.attn(self.ln1(z))
        z = z + self.mlp(self.ln2(z))
        return z

# ---- Main module plugged into BERT ----------------------------------------
class LMABertAttention(torch.nn.Module):
    """
    Latent Meta Attention module for BERT.
    Steps   : 1) head‑stacking, 2) re‑chunking to latent '[L_new,C_new]',
              3) Linear → d_new, 4) N latent blocks, 5) Linear → C_new,
              6) inverse re‑chunk & un‑stack back to '[B,S,H]'.
    Config  : supply the following attributes inside cfg_attention
              ─ num_heads_stacking  (int, default 4)
              ─ target_l_new        (int, optional, default S//2)
              ─ d_new               (int, default H//2)
              ─ num_heads_latent    (int, default 4)
              ─ ff_latent_hidden    (int, default 4*d_new)
              ─ num_lma_layers      (int, default 2)
              ─ dropout_prob, qkv_bias (already present)
    """
    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size: int, cfg_attention):
        super().__init__()
        # hyper‑parameters from cfg_attention
        self.hidden_size        = hidden_size
        self.nh_stack           = int(getattr(cfg_attention, "num_heads_stacking", 4))
        self.d_new              = int(getattr(cfg_attention, "d_new", hidden_size // 2))
        self.nh_latent          = int(getattr(cfg_attention, "num_heads_latent", 4))
        self.ff_latent_hidden   = int(getattr(cfg_attention, "ff_latent_hidden", 4 * self.d_new))
        self.n_layers           = int(getattr(cfg_attention, "num_lma_layers", 2))
        self.target_l_new_cfg   = getattr(cfg_attention, "target_l_new", None)
        self.dropout            = cfg_attention.dropout_prob
        self.bias               = cfg_attention.qkv_bias

        if hidden_size % self.nh_stack != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by num_heads_stacking {self.nh_stack}")

        # modules built lazily because they depend on seq_len / C_new
        self._built          = False
        self.output_dim      = hidden_size  # required by AttentionComponent

    # ---------------------------------------------------------------------
    def _build(self, seq_len: int, device: torch.device):
        """Create sub‑modules for a fixed sequence length."""
        dk              = self.hidden_size // self.nh_stack          # per‑head dim after stacking
        total_features  = seq_len * self.hidden_size                 # S*H
        target_l_new    = (self.target_l_new_cfg
                           if self.target_l_new_cfg is not None
                           else max(2, seq_len // 2))
        self.L_new      = _find_closest_divisor(total_features, target_l_new)
        self.C_new      = total_features // self.L_new

        # linear projections C_new ↔ d_new
        self.to_latent  = torch.nn.Linear(self.C_new, self.d_new,  bias=self.bias)
        self.from_latent= torch.nn.Linear(self.d_new, self.C_new,  bias=self.bias)
        # --- ensure lazy‑constructed parameters live on the same device as the incoming tensor ---
        self.to_latent  = self.to_latent.to(device)
        self.from_latent= self.from_latent.to(device)

        # latent transformer blocks
        self.blocks = torch.nn.ModuleList([
            _LMABlock(self.d_new, self.nh_latent, self.ff_latent_hidden,
                      self.dropout, self.bias)
            for _ in range(self.n_layers)
        ])
        # move latent blocks to the correct device
        self.blocks = self.blocks.to(device)

        # cache constants for the inverse un‑stacking
        self.dk           = dk
        self.seq_len      = seq_len
        self.register_buffer("_dummy", torch.empty(0, device=device))  # to track device
        self._built       = True

    # ---------------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None):
        """
        hidden_states : Tensor [B,S,H]
        attention_mask is ignored — masking would require mapping the mask
        through the same reshaping steps.
        """
        B, S, H = hidden_states.shape
        if not self._built:
            self._build(S, hidden_states.device)
        elif S != self.seq_len:
            raise RuntimeError(f"LMABertAttention was built for seq_len={self.seq_len} "
                               f"but got {S}")

        # --- Stage 2a : head‑stacking --------------------------------------
        dk = self.dk
        head_views  = torch.split(hidden_states, dk, dim=2)      # list len=nh_stack
        x_stacked   = torch.cat(head_views, dim=1)               # [B,S*nh_stack,dk]

        # --- Stage 2b : re‑chunk & project to latent ----------------------
        flat        = x_stacked.view(B, -1)                      # [B,S*H]
        x_chunks    = flat.view(B, self.L_new, self.C_new)       # [B,L_new,C_new]
        z           = self.to_latent(x_chunks)                   # [B,L_new,d_new]

        # --- Latent transformer ------------------------------------------
        for blk in self.blocks:
            z = blk(z)

        # --- Project back & inverse reshape ------------------------------
        chunks_back = self.from_latent(z)                        # [B,L_new,C_new]
        flat_back   = chunks_back.reshape(B, -1)                 # [B,S*H]
        x_stacked_b = flat_back.view(B, S * self.nh_stack, dk)   # [B,S*nh,dk]

        # inverse head stacking
        x_unstacked = x_stacked_b.view(B, self.seq_len, self.nh_stack, dk)
        out         = torch.cat(torch.unbind(x_unstacked, dim=2), dim=2)  # [B,S,H]
        return out


def get_attention_mechanism(
    idx,
    hidden_size,
    cfg_attention,
):
    if cfg_attention.type == "self-attention":
        mechanism = SeqFirstSelfAttention(hidden_size, cfg_attention)  # neox
    elif cfg_attention.type == "pytorch":
        # Sanity check 1: [Warning: This includes the output projection twice...]
        mechanism = SelfAttentionPyTorch(hidden_size, cfg_attention)  # torch default
    elif cfg_attention.type == "pytorch-seqfirst":
        # Sanity check 1: [Warning: This includes the output projection twice...]
        mechanism = SeqFirstSelfAttentionPyTorch(hidden_size, cfg_attention)  # torch default
    elif cfg_attention.type == "huggingface":
        mechanism = BertAttentionWrapper(hidden_size, cfg_attention)  # always includes bias!
    elif cfg_attention.type == "flash-attention-impl":  # the fast implementation called flash
        mechanism = FlashMultiHeadAttention(hidden_size, cfg_attention)
    elif cfg_attention.type == "fourier":
        mechanism = FourierMixing(hidden_size, cfg_attention)
    elif cfg_attention.type == "fourier-experimental":
        mechanism = FourierMixingParametrized(hidden_size, cfg_attention)
    elif cfg_attention.type == "flash":  # flash from transformer quality in linear time
        mechanism = FLASH(hidden_size, cfg_attention)
    elif cfg_attention.type == "tuformer":
        mechanism = TuFormAttention(hidden_size, cfg_attention)
    elif cfg_attention.type == "funnel":  # dont use this with a normal seq->seq model
        mechanism = FunnelAttention(hidden_size, cfg_attention)
    elif cfg_attention.type == "seqfirst_tuformer":
        mechanism = SeqFirstTuFormAttention(hidden_size, cfg_attention)
    elif cfg_attention.type == "seqfirst2_tuformer":
        mechanism = SeqFirstTuFormAttention(hidden_size, cfg_attention)
    elif cfg_attention.type == "none":
        mechanism = Identity(hidden_size)
    elif cfg_attention.type == "fourier-hybrid":
        if idx in cfg_attention.hybrid_layers:
            mechanism = SeqFirstSelfAttention(hidden_size, cfg_attention)
        else:
            mechanism = FourierMixing(hidden_size, cfg_attention)
    elif cfg_attention.type == "lma":
        mechanism = LMABertAttention(hidden_size, cfg_attention)
    else:
        raise ValueError(f"Invalid attention type {cfg_attention.type} given.")
    return mechanism


class Identity(torch.nn.Module):
    """mini wrapper around BERT attention from huggingface for sanity checks."""

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size):
        super().__init__()
        self.output_dim = hidden_size

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return hidden_states


class BertAttentionWrapper(BertSelfAttention):
    """mini wrapper around BERT attention from huggingface for sanity checks."""

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size, cfg_attention):
        class config:
            pass

        config.hidden_size = hidden_size
        config.num_attention_heads = cfg_attention.num_attention_heads
        config.attention_probs_dropout_prob = cfg_attention.dropout_prob
        config.is_decoder = False

        super().__init__(config)
        self.output_dim = hidden_size

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return super().forward(hidden_states, attention_mask)[0]


class SelfAttentionPyTorch(torch.nn.Module):
    """Minimal wrapper around pytorch self attention."""

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size, cfg_attention):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            hidden_size,
            cfg_attention.num_attention_heads,
            dropout=cfg_attention.dropout_prob,
            batch_first=True,
            bias=False,
            add_bias_kv=cfg_attention.qkv_bias,
        )

        # Do something terrible to patch the fact that the output projection is somewhere else in our code:
        del self.attn.out_proj.weight
        del self.attn.out_proj.bias
        self.attn.out_proj.register_buffer("weight", torch.eye(hidden_size))
        self.attn.out_proj.register_buffer("bias", torch.zeros(hidden_size))
        self.output_dim = hidden_size

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return self.attn(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask, need_weights=False)[0]


class SeqFirstSelfAttentionPyTorch(torch.nn.Module):
    """Minimal wrapper around pytorch self attention."""

    __constants__ = ["LAYOUT"]
    LAYOUT = "[S B H]"

    def __init__(self, hidden_size, cfg_attention):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            hidden_size,
            cfg_attention.num_attention_heads,
            dropout=cfg_attention.dropout_prob,
            batch_first=False,
            bias=False,
            add_bias_kv=cfg_attention.qkv_bias,
        )

        # Do something terrible to patch the fact that the output projection is somewhere else in our code:
        del self.attn.out_proj.weight
        del self.attn.out_proj.bias
        self.attn.out_proj.register_buffer("weight", torch.eye(hidden_size))
        self.attn.out_proj.register_buffer("bias", torch.zeros(hidden_size))
        self.output_dim = hidden_size

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return self.attn(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask, need_weights=False)[0]


class LegacySeqFirstSelfAttention(torch.nn.Module):
    """Self-attention layer.

    This is the gpt neo-x implementation from:
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py (which is a megatron variant)

    Self-attention layer takes input with size [Seq, Batch, Hidden]
    and returns output of the same size.
    """

    __constants__ = ["LAYOUT", "attention_dropout"]
    LAYOUT: str = "[S B H]"
    norm_factor: torch.Tensor

    def __init__(self, hidden_size: int, cfg_attention):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = cfg_attention.num_attention_heads
        self.hidden_per_head = self.hidden_size // cfg_attention.num_attention_heads
        self.register_buffer("norm_factor", torch.tensor(self.hidden_per_head).rsqrt())

        # Strided linear layer.
        self.query_key_value = torch.nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=cfg_attention.qkv_bias)
        self.output_dim = hidden_size
        if cfg_attention.rotary_embedding == "sanity":
            self.rotary_emb = RotarySanityCheck(self.hidden_per_head, seq_dim=0)
        elif cfg_attention.rotary_embedding == "v2":
            self.rotary_emb = RotaryEleutherAI(self.hidden_per_head)
        elif cfg_attention.rotary_embedding == "llama":
            self.rotary_emb = RotaryLLAMA(self.hidden_per_head)
        elif cfg_attention.rotary_embedding:
            self.rotary_emb = Rotary(self.hidden_per_head, seq_dim=0)
        else:
            self.rotary_emb = None

        if cfg_attention.sequence_op == "torch-softmax":
            self.sequence_op = TorchSoftmax(cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "torch-norm":
            self.sequence_op = TorchNormalize(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "none":
            self.sequence_op = ScaledIdentity(cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "cumsum":
            self.sequence_op = Cumsum(cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "cumsumexp":
            self.sequence_op = CumsumExp(cfg_attention.seq_op_in_fp32)
        else:
            raise ValueError(f"Invalid sequence operation {cfg_attention.sequence_op} given.")

        self.attention_dropout: float = cfg_attention.dropout_prob

    def attention(self, query_layer, key_layer, value_layer, attention_mask: Optional[torch.Tensor] = None, training: bool = False):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )  # this looks crazy but beta=0 below skips the values of this tensor [so beta is NOT optional...]

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=self.norm_factor,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(output_size[0], output_size[1], output_size[2], output_size[3])

        # ===========================
        # Attention probs and dropout
        # ===========================
        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.sequence_op(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # [And in great ML tradition I keep this comment in place as it was in megatron and huggingface-bert before :>]
        # attention_probs = self.attention_dropout(attention_probs)
        attention_probs = torch.nn.functional.dropout(attention_probs, p=self.attention_dropout, training=training)
        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        # =====================
        # hidden_states: [sq, b, h]
        # Query, Key, and Value
        # =====================
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        # new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads, 3 * self.hidden_per_head)
        mixed_x_layer = mixed_x_layer.view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, 3 * self.hidden_per_head
        )

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [self.hidden_per_head] * 3, dim=3)

        if self.rotary_emb is not None:
            query_layer, key_layer = self.rotary_emb(query_layer, key_layer)

        # ==================================
        # Attention computation
        # ==================================
        context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask, self.training)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        # new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], self.hidden_size)
        return context_layer


class SeqFirstSelfAttention(LegacySeqFirstSelfAttention):
    """Self-attention layer.

    This is the gpt neo-x implementation from:
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py (which is a megatron variant)

    This is a modified version of the neo-x implementation that I can manage to compile without graph breaks
    """

    __constants__ = ["LAYOUT", "attention_dropout"]
    LAYOUT: str = "[S B H]"
    norm_factor: torch.Tensor

    def attention(self, query_layer, key_layer, value_layer, attention_mask: Optional[torch.Tensor] = None, training: bool = False):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # this better be fused in a clever way:
        matmul_result = torch.bmm(query_layer.transpose(0, 1), key_layer.transpose(0, 1).transpose(1, 2)) * self.norm_factor

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(output_size[0], output_size[1], output_size[2], output_size[3])

        # ===========================
        # Attention probs and dropout
        # ===========================
        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.sequence_op(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # [And in great ML tradition I keep this comment in place as it was in megatron and huggingface-bert before :>]
        # attention_probs = self.attention_dropout(attention_probs)
        attention_probs = torch.nn.functional.dropout(attention_probs, p=self.attention_dropout, training=training)
        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer


class FlashMultiHeadAttention(torch.nn.Module):
    """Wrapper for flash MHA."""

    __constants__ = ["LAYOUT"]
    LAYOUT: str = "[B S H]"

    def __init__(self, hidden_size, cfg_attention):
        super().__init__()
        from flash_attn.flash_attention import FlashMHA

        self.flash_mha = FlashMHA(
            hidden_size,
            cfg_attention.num_attention_heads,
            bias=cfg_attention.qkv_bias,
            batch_first=True,
            attention_dropout=cfg_attention.dropout_prob,
            causal=cfg_attention.causal_attention,
        )
        hidden_per_head = hidden_size // self.flash_mha.num_heads
        if cfg_attention.rotary_embedding:
            if cfg_attention.low_level_fusion:
                self.rotary_emb = torch.jit.script(Rotary(hidden_per_head, seq_dim=1))
            else:
                self.rotary_emb = Rotary(hidden_per_head, seq_dim=1)
        else:
            self.rotary_emb = None

        self.flash_mha.out_proj = None
        self.output_dim = hidden_size

    @torch.jit.ignore  # This jit.ignore call is ignored?
    def flash_inner(self, qkv):
        return self.flash_mha.inner_attn(qkv, key_padding_mask=None, need_weights=False, causal=self.flash_mha.causal)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        # def forward(self, x, key_padding_mask=None, need_weights=False):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)

        Returns only the rearranged, unprojected output
        """
        qkv = self.flash_mha.Wqkv(hidden_states)
        if self.rotary_emb is not None:
            query, key, value = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.flash_mha.num_heads).unbind(dim=2)
            query, key = self.rotary_emb(query, key)
            qkv = torch.stack([query.type(qkv.dtype), key.type(qkv.dtype), value.type(qkv.dtype)], dim=2)
        else:
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.flash_mha.num_heads)
        context, attn_weights = self.flash_inner(qkv)
        return rearrange(context, "b s h d -> b s (h d)")


class FunnelAttention(SeqFirstSelfAttention):
    """Self-attention layer abstract class.

    This is a funnel crammed into the gpt neo-x implementation from:
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py (which is a megatron variant)

    Self-attention layer takes input with size [Seq, Batch, Hidden]
    and returns output of the same size.
    """

    __constants__ = ["LAYOUT", "attention_dropout", "length_factor"]
    LAYOUT: str = "[S B H]"
    norm_factor: torch.Tensor

    def __init__(self, hidden_size, cfg_attention, length_factor=1.0):
        super().__init__(hidden_size, cfg_attention)
        self.length_factor: float = length_factor

        # Strided linear layers
        del self.query_key_value
        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        self.key_value = torch.nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=cfg_attention.qkv_bias)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        # =====================
        # hidden_states: [sq, b, h]
        # Query, Key, and Value
        # =====================

        # ==================================
        #  Pool or unpool states
        # ==================================
        sq, b = hidden_states.shape[0], hidden_states.shape[1]

        # [sq, b, h] -> [sq * F, b, h]
        new_seq_length = int(sq * self.length_factor)
        if self.length_factor < 1:
            query_states = hidden_states.view(int(1 / self.length_factor), new_seq_length, b, self.hidden_size).mean(dim=0)
        elif self.length_factor > 1:
            query_states = hidden_states.repeat_interleave(int(self.length_factor), dim=0, output_size=new_seq_length)
        else:
            query_states = hidden_states

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        query_layer = self.query(query_states).view(new_seq_length, b, self.num_attention_heads, self.hidden_per_head)
        mixed_x_layer = self.key_value(hidden_states).view(sq, b, self.num_attention_heads, 2 * self.hidden_per_head)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (key_layer, value_layer) = torch.split(mixed_x_layer, [self.hidden_per_head] * 2, dim=3)

        if self.rotary_emb is not None:
            query_layer = self.rotary_emb.single_forward(query_layer)
            key_layer = self.rotary_emb.single_forward(key_layer)

        # ==================================
        # Attention computation
        # ==================================
        context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask, self.training)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        # new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(new_seq_length, context_layer.shape[1], self.hidden_size)
        return context_layer


class TuFormAttention(torch.nn.Module):
    """Self-attention layer abstract class.

    This is a simplification of the tuformer implementationfrom
    https://github.com/xliu1231/fairseq_tuformer/blob/main/fairseq/modules/tuckerhead_attention.py

    THSA layer takes input with size [Batch, Seq, Hidden]
    and returns output of the same size.
    """

    __constants__ = ["LAYOUT"]
    LAYOUT: str = "[B S H]"

    def __init__(self, hidden_size, cfg_attention):
        super().__init__()
        self.hidden_size = hidden_size
        self.rdim = getattr(cfg_attention, "rdim", hidden_size)
        self.register_buffer("norm_factor", torch.tensor(self.rdim).rsqrt())

        # Strided linear layer.
        self.query_key_value = torch.nn.Linear(self.hidden_size, 3 * self.rdim, bias=cfg_attention.qkv_bias)
        self.c_proj = torch.nn.Linear(self.rdim, self.rdim, bias=cfg_attention.qkv_bias)
        self.output_dim = self.rdim

        if cfg_attention.rotary_embedding:
            raise ValueError("Have to think about dimensions here.")

        if cfg_attention.sequence_op == "torch-softmax":
            self.sequence_op = torch.jit.script(TorchSoftmax(cfg_attention.seq_op_in_fp32))
        elif cfg_attention.sequence_op == "torch-norm":
            self.sequence_op = torch.jit.script(TorchNormalize(self.num_attention_heads, cfg_attention.seq_op_in_fp32))
        elif cfg_attention.sequence_op == "none":
            self.sequence_op = torch.jit.script(ScaledIdentity(cfg_attention.seq_op_in_fp32))
        elif cfg_attention.sequence_op == "cumsum":
            self.sequence_op = torch.jit.script(Cumsum(cfg_attention.seq_op_in_fp32))
        elif cfg_attention.sequence_op == "cumsumexp":
            self.sequence_op = torch.jit.script(CumsumExp(cfg_attention.seq_op_in_fp32))
        else:
            raise ValueError(f"Invalid sequence operation {cfg_attention.sequence_op} given.")

        self.attention_dropout = torch.nn.Dropout(cfg_attention.dropout_prob, inplace=False)  # cannot be inplace
        self.first_rearrange = Rearrange("b s l r -> (b r) s l", r=self.rdim)
        self.second_rearrange = Rearrange("(b r) s l -> b r s l", r=self.rdim)

    def attention(self, query_layer, key_layer, value_layer, attention_mask: Optional[torch.Tensor] = None):
        """tuformer attention in batch first implementation (hopefully)"""
        attention_scores = self.c_proj(torch.einsum("bsr, blr -> bslr", query_layer, key_layer))

        attention_scores = self.sequence_op(self.first_rearrange(attention_scores), attention_mask)
        attention_scores = self.attention_dropout(attention_scores)

        return torch.einsum("brsl, blr -> bsr", self.second_rearrange(attention_scores), value_layer)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        mixed_x_layer = self.query_key_value(hidden_states)  # b s 3r

        # 3 [ b s r]
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [self.rdim] * 3, dim=-1)
        context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask)  # b s r
        return context_layer


class SeqFirstTuFormAttention(TuFormAttention):
    """Self-attention layer abstract class.

    Seq-first variant 1

    THSA layer takes input with size [Seq, Batch, Hidden]
    and returns output of the same size.
    """

    __constants__ = ["LAYOUT"]
    LAYOUT: str = "[S B H]"

    def __init__(
        self,
        hidden_size,
        cfg_attention,
    ):
        super().__init__(hidden_size, cfg_attention)
        self.first_rearrange = Rearrange("b s l r -> (b r) s l", r=self.rdim)
        self.second_rearrange = Rearrange("(b r) s l -> b r s l", r=self.rdim)

    def attention(self, query_layer, key_layer, value_layer, attention_mask: Optional[torch.Tensor] = None):
        """tuformer attention in batch first implementation (hopefully)"""
        attention_scores = self.c_proj(torch.einsum("sbr, lbr -> bslr", query_layer, key_layer))

        attention_scores = self.sequence_op(self.first_rearrange(attention_scores), attention_mask)
        attention_scores = self.attention_dropout(attention_scores)
        return torch.einsum("brsl, lbr -> sbr", self.second_rearrange(attention_scores), value_layer)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        mixed_x_layer = self.query_key_value(hidden_states)  # b s 3r

        # 3 [ b s r]
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [self.rdim] * 3, dim=-1)
        context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask)  # b s r
        return context_layer


class SeqFirstTuFormAttention2(TuFormAttention):
    """Self-attention layer abstract class.

    Seq-first variant 2

    THSA layer takes input with size [Seq, Batch, Hidden]
    and returns output of the same size.
    """

    __constants__ = ["LAYOUT"]
    LAYOUT: str = "[S B H]"

    def __init__(
        self,
        hidden_size,
        cfg_attention,
    ):
        super().__init__(hidden_size, cfg_attention)
        self.first_rearrange = Rearrange("s l b r -> s l (b r)", r=self.rdim)
        self.second_rearrange = Rearrange("s l (b r) -> s l b r", r=self.rdim)
        if cfg_attention.sequence_op != "torch-softmax":
            raise ValueError("Not implemented")

    def attention(self, query_layer, key_layer, value_layer, attention_mask: Optional[torch.Tensor] = None):
        """tuformer attention in batch first implementation (hopefully)"""
        attention_scores = self.c_proj(torch.einsum("sbr, lbr -> slbr", query_layer, key_layer))

        attention_scores = self.first_rearrange(attention_scores).softmax(dim=1)
        attention_scores = self.attention_dropout(attention_scores)
        return torch.einsum("slbr, lbr -> sbr", self.second_rearrange(attention_scores), value_layer)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        mixed_x_layer = self.query_key_value(hidden_states)  # b s 3r

        # 3 [ b s r]
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [self.rdim] * 3, dim=-1)
        context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask)  # b s r
        return context_layer


class FourierMixing(torch.nn.Module):
    """Fourier mixing layer as described in the FNet paper.
    Layer takes input with size [Batch, Seq, Hidden] and returns output of the same size.
    This function can take an attention mask as input, but will ignore it.
    """

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size, cfg_attention):
        super().__init__()
        self.fft_op_in_fp32 = True  # Always necessary (atleast on pytorch 1.12)
        self.output_dim = hidden_size
        if cfg_attention.rotary_embedding:
            if cfg_attention.low_level_fusion:
                self.rotary_emb = torch.jit.script(Rotary(hidden_size, seq_dim=1))
            else:
                self.rotary_emb = Rotary(hidden_size, seq_dim=0)
        else:
            self.rotary_emb = None

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        """Forward will take an attention mask but ignore it!"""

        if self.rotary_emb is not None:
            # full rotary (mostly on for compatibility, no guarantees on this being non-terrible)
            cos, sin = self.rotary_emb.get_cos_sin_cache(hidden_states)
            hidden_states = (hidden_states * cos[:, 0]) + (self.rotary_emb.rotate_half(hidden_states) * sin[:, 0])

        if self.fft_op_in_fp32:
            hidden_state_dtype = hidden_states.dtype
            hidden_states = hidden_states.float()
        else:
            hidden_state_dtype = None

        # Implementation 1:
        # hidden_states = torch.fft.fft(torch.fft.fft(hidden_states, dim=0, , norm="ortho"), dim=2, , norm="ortho").real
        # Implementation 2:
        hidden_states = torch.fft.fftn(hidden_states, dim=(1, 2), norm="ortho").real  # could also cast into angle?

        if self.fft_op_in_fp32:
            hidden_states = hidden_states.to(hidden_state_dtype)

        return hidden_states


class FourierMixingParametrized(torch.nn.Module):
    """Fourier mixing layer as described in the FNet paper.
    Layer takes input with size [Seq, batch, Hidden] and returns output of the same size.
    This function can take an attention mask as input, but will ignore it.
    """

    __constants__ = ["LAYOUT"]
    LAYOUT = "[S B H]"

    def __init__(
        self,
        hidden_size,
        cfg_attention,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = cfg_attention.num_attention_heads
        self.hidden_per_head = self.hidden_size // cfg_attention.num_attention_heads
        self.fft_op_in_fp32 = True  # Always necessary (atleast on pytorch 1.12)

        # linear layer.
        self.projection = torch.nn.Linear(2 * self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        self.output_dim = hidden_size
        if cfg_attention.rotary_embedding:
            if cfg_attention.low_level_fusion:
                self.rotary_emb = torch.jit.script(Rotary(self.hidden_per_head, seq_dim=0))
            else:
                self.rotary_emb = Rotary(self.hidden_per_head, seq_dim=0)
        else:
            self.rotary_emb = None

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        """Forward will take an attention mask but ignore it!"""

        # [S, B, (np * hn)] --> [S, B, np, hn]
        head_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
        if self.rotary_emb is not None:
            # full rotary (mostly on for compatibility, no guarantees on this being non-terrible)
            cos, sin = self.rotary_emb.get_cos_sin_cache(head_states)
            hidden_states = (head_states * cos[:, 0]) + (self.rotary_emb.rotate_half(head_states) * sin[:, 0])

        if self.fft_op_in_fp32:
            hidden_state_dtype = hidden_states.dtype
            head_states = head_states.float()
        else:
            hidden_state_dtype = None

        # Implementation 2:
        complex_scores = torch.fft.fftn(head_states, dim=(2, 3), norm="ortho")
        # complex [S, B, np, hn] -> [S, B, 2 * np * hn]
        # need to restride for this :<
        head_states = torch.view_as_real(complex_scores).reshape(hidden_states.shape[0], hidden_states.shape[1], -1)

        if self.fft_op_in_fp32:
            head_states = head_states.to(hidden_state_dtype)

        hidden_states = self.projection(head_states)

        return hidden_states


class FLASH(torch.nn.Module):
    """FLASH as described in Transformer Quality in Linear Time.
    This is FLASH-QUAD, as we're not too interested in long-range sequences here.
    """

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size, cfg_attention, expansion_factor: int = 2, s: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.e = hidden_size * expansion_factor
        self.s = s
        self.uv_projection = torch.nn.Linear(hidden_size, 2 * self.e + self.s, bias=cfg_attention.qkv_bias)
        self.nonlin = torch.nn.SiLU(inplace=False)
        self.gamma = torch.nn.Parameter(torch.randn(2, s) * 0.02)
        self.beta = torch.nn.Parameter(torch.zeros(2, s))

        self.out_projection = torch.nn.Linear(self.e, hidden_size, bias=cfg_attention.qkv_bias)
        self.output_dim = hidden_size

        if cfg_attention.rotary_embedding:
            if cfg_attention.low_level_fusion:
                self.rotary_emb = torch.jit.script(Rotary(self.s, seq_dim=1))
            else:
                self.rotary_emb = Rotary(self.s, seq_dim=1)
        else:
            self.rotary_emb = None

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        """Inputs of shape [B, S, H]. Implementation directly based on FLASH pseudocode (see paper appendix)"""
        u_v_base = self.nonlin(self.uv_projection(inputs))
        u, v, base = torch.split(u_v_base, [self.e, self.e, self.s], dim=-1)
        base = torch.einsum("...r,hr->...hr", base, self.gamma) + self.beta
        if self.rotary_emb is not None:
            base = self.rotary_emb.single_forward(base)
        query, key = torch.unbind(base, dim=2)

        attention_scores = query.matmul(key.transpose(1, 2)) / inputs.shape[1]
        squared_scores = torch.nn.functional.relu(attention_scores).pow(2)
        return self.out_projection(u * torch.einsum(" bnm,bme->bne ", squared_scores, v))


class TorchSoftmax(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, seq_op_in_fp32=False):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        if attention_mask is not None:
            inputs = inputs + attention_mask
        probs = torch.softmax(inputs, dim=-1).to(dtype=input_dtype)
        return probs


class TorchNormalize(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, seq_op_in_fp32=False):
        """Normalized attention pooling as described in Richter&Wattenhofer, 2020."""
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32
        self.seq_gamma = torch.nn.Parameter(torch.ones(1, num_attention_heads, 1, 1))
        self.seq_beta = torch.nn.Parameter(torch.zeros(1, num_attention_heads, 1, 1))

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        # Inputs are [b, np, sq, sk]
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)

        if attention_mask is not None:
            inputs[attention_mask != 0] = 0

        norms = torch.nn.functional.layer_norm(inputs, inputs.shape[1:], eps=1e-05)
        norms = (norms * self.seq_gamma + self.seq_beta).to(dtype=input_dtype)
        return norms


class ScaledIdentity(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, seq_op_in_fp32):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        """Sequence-scaled input."""
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
        return (inputs * torch.as_tensor(inputs.shape[2]).rsqrt()).to(dtype=input_dtype)


class Cumsum(torch.nn.Module):
    def __init__(self, seq_op_in_fp32):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        """Sequence-scaled input cumulative sum."""
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
        return (inputs.cumsum(dim=-1) * pow(inputs.shape[2], -0.5)).to(dtype=input_dtype)


class CumsumExp(torch.nn.Module):
    def __init__(self, seq_op_in_fp32):
        super().__init__()
        self.seq_op_in_fp32 = True  # Required as of pytorch 1.13

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        """Sequence-scaled input cumulative sum."""
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
        return (inputs.logcumsumexp(dim=-1) * pow(inputs.shape[2], -0.5)).to(dtype=input_dtype)


# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# import math
# from dataclasses import dataclass, field
# from matplotlib.animation import FuncAnimation
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# #===============================================
# # LMA Feature Extractor Implementation
# #===============================================

# @dataclass
# class LMAConfigRL:
#     """ Config for LMA Feature Extractor """
#     seq_len: int             # Input sequence length (L, e.g., 6)
#     embed_dim: int           # Initial embedding dim (d0)
#     num_heads_stacking: int  # Heads for stacking (nh)
#     target_l_new: int        # Target latent sequence length
#     d_new: int               # Latent embedding dim
#     num_heads_latent: int    # Heads for latent attention

#     # Derived values
#     L_new: int = field(init=False) # Actual latent sequence length
#     C_new: int = field(init=False) # Latent chunk size

#     def __post_init__(self):
#         if self.seq_len <= 0 or self.embed_dim <= 0 or self.num_heads_stacking <= 0 or \
#            self.target_l_new <= 0 or self.d_new <= 0 or self.num_heads_latent <= 0:
#             raise ValueError("LMAConfigRL inputs must be positive.")
#         if self.embed_dim % self.num_heads_stacking != 0:
#             raise ValueError(f"LMA embed_dim ({self.embed_dim}) not divisible by num_heads_stacking ({self.num_heads_stacking})")
#         if self.d_new % self.num_heads_latent != 0:
#             raise ValueError(f"LMA d_new ({self.d_new}) not divisible by num_heads_latent ({self.num_heads_latent})")

#         total_features = self.seq_len * self.embed_dim
#         if total_features == 0: raise ValueError("LMA total features cannot be zero.")

#         try:
#             self.L_new = find_closest_divisor(total_features, self.target_l_new)
#             if self.L_new != self.target_l_new:
#                  print(f"LMAConfigRL ADJUSTMENT: L_new {self.target_l_new} -> {self.L_new}")
#             if self.L_new <= 0: raise ValueError("Calculated L_new is not positive.")
#             if total_features % self.L_new != 0:
#                 raise RuntimeError(f"Internal Error: total_features ({total_features}) not divisible by final L_new ({self.L_new})")
#             self.C_new = total_features // self.L_new
#             if self.C_new <= 0: raise ValueError("Calculated C_new is not positive.")
#         except ValueError as e:
#             raise ValueError(f"LMA Config Error calculating L_new/C_new: {e}") from e

# class LayerNorm(nn.Module):
#     """ LayerNorm with optional bias """
#     def __init__(self, ndim, bias=True):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(ndim))
#         self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
#     def forward(self, input):
#         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# class LMA_InitialTransform_RL(nn.Module):
#     """ Performs LMA Stage 1 and Stage 2 (Stacking, Rechunking, Latent Embed) """
#     def __init__(self, features_per_step: int, lma_config: LMAConfigRL, dropout: float, bias: bool):
#         super().__init__()
#         self.lma_config = lma_config
#         self.dropout_p = dropout
#         self.bias = bias

#         # Stage 1 equivalent: Project features per step to embed_dim (d0)
#         self.input_embedding = nn.Linear(features_per_step, lma_config.embed_dim, bias=self.bias)
#         self.input_embedding_act = nn.ReLU() # Activation defined
#         self.embedding_dropout = nn.Dropout(p=self.dropout_p)

#         # Stage 2b: Latent Embedding Layer (maps C_new -> d_new)
#         self.embed_layer_2 = nn.Linear(lma_config.C_new, lma_config.d_new, bias=self.bias)
#         # Using ReLU here as requested (instead of GELU in prev version)
#         self.embed_layer_2_act = nn.ReLU()

#         print("  LMA_InitialTransform_RL Initialized:")
#         print(f"    Input features/step: {features_per_step}")
#         print(f"    Stage 1 Projection: Linear({features_per_step} -> {lma_config.embed_dim}) + ReLU") # Updated print
#         print(f"    Head Stacking: {lma_config.num_heads_stacking} heads")
#         print(f"    Rechunking: L={lma_config.seq_len}, d0={lma_config.embed_dim} -> L_new={lma_config.L_new}, C_new={lma_config.C_new}")
#         print(f"    Stage 2b Projection: Linear({lma_config.C_new} -> {lma_config.d_new}) + ReLU") # Updated print

#     def forward(self, x):
#         # Input x shape: (B, L, features_per_step)
#         B, L, _ = x.shape
#         if L != self.lma_config.seq_len:
#             raise ValueError(f"Input sequence length {L} doesn't match LMA config seq_len {self.lma_config.seq_len}")

#         # --- Stage 1 ---
#         y = self.input_embedding(x) # (B, L, Feat/L) -> (B, L, d0)
#         y = self.input_embedding_act(y) # *** APPLY ACTIVATION HERE ***
#         y = y + self._positional_encoding(L, self.lma_config.embed_dim).to(y.device)
#         y = self.embedding_dropout(y) # (B, L, d0)

#         # --- Stage 2a: Head-View Stacking ---
#         d0 = self.lma_config.embed_dim
#         nh = self.lma_config.num_heads_stacking
#         dk = d0 // nh
#         try:
#             head_views = torch.split(y, dk, dim=2)
#             x_stacked = torch.cat(head_views, dim=1) # (B, L*nh, dk)
#         except Exception as e:
#             raise RuntimeError(f"Error during head stacking: Input={y.shape}, d0={d0}, nh={nh}, dk={dk}") from e

#         # --- Stage 2b: Re-Chunking & Latent Embedding ---
#         L_new = self.lma_config.L_new
#         C_new = self.lma_config.C_new
#         expected_flat_dim = L * d0

#         x_flat = x_stacked.view(B, -1) # (B, L*d0)
#         if x_flat.shape[1] != expected_flat_dim:
#              raise RuntimeError(f"Flattened shape mismatch: Expected {expected_flat_dim}, got {x_flat.shape[1]}")

#         try: x_rechunked = x_flat.view(B, L_new, C_new) # (B, L_new, C_new)
#         except RuntimeError as e: raise RuntimeError(f"Error rechunking: Flat={x_flat.shape}, Target=({B}, {L_new}, {C_new})") from e

#         z_embedded = self.embed_layer_2(x_rechunked) # (B, L_new, d_new)
#         z = self.embed_layer_2_act(z_embedded) # Apply activation

#         return z # Return latent representation

#     def _positional_encoding(self, seq_len, embed_dim):
#         # (Keep positional encoding function as before)
#         position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
#         pe = torch.zeros(seq_len, embed_dim)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         return pe

# class LatentAttention_RL(nn.Module):
#     """ MHA operating in the LMA latent space (Non-Causal) """
#     def __init__(self, d_new: int, num_heads_latent: int, dropout: float, bias: bool):
#         super().__init__()
#         assert d_new % num_heads_latent == 0
#         self.d_new = d_new
#         self.num_heads = num_heads_latent
#         self.head_dim = d_new // num_heads_latent
#         self.dropout_p = dropout
#         self.bias = bias

#         self.c_attn = nn.Linear(d_new, 3 * d_new, bias=self.bias)
#         self.c_proj = nn.Linear(d_new, d_new, bias=self.bias)
#         self.attn_dropout = nn.Dropout(self.dropout_p)
#         self.resid_dropout = nn.Dropout(self.dropout_p)
#         self.flash = hasattr(F, 'scaled_dot_product_attention')
#         if self.flash: print(f"    - LatentAttention_RL: Using Flash Attention (d_new={d_new})")
#         else: print(f"    - LatentAttention_RL: Using slow attention path.")

#     def forward(self, z):
#         B, L_new, C = z.size()
#         if C != self.d_new: raise ValueError(f"LatentAttention C mismatch")
#         q, k, v = self.c_attn(z).split(self.d_new, dim=2)
#         q = q.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)
#         k = k.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)
#         v = v.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)
#         if self.flash:
#             y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.training else 0, is_causal=False)
#         else:
#             att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
#             att = F.softmax(att, dim=-1)
#             att = self.attn_dropout(att)
#             y = att @ v
#         y = y.transpose(1, 2).contiguous().view(B, L_new, self.d_new)
#         y = self.resid_dropout(self.c_proj(y))
#         return y

# class LatentMLP_RL(nn.Module):
#     """ MLP operating in the latent space dimension d_new """
#     def __init__(self, d_new: int, ff_latent_hidden: int, dropout: float, bias: bool):
#         super().__init__()
#         self.c_fc    = nn.Linear(d_new, ff_latent_hidden, bias=bias)
#         self.gelu    = nn.GELU()
#         self.c_proj  = nn.Linear(ff_latent_hidden, d_new, bias=bias)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.c_fc(x); x = self.gelu(x); x = self.c_proj(x); x = self.dropout(x)
#         return x

# class LMABlock_RL(nn.Module):
#     """ A single LMA block operating in the latent space """
#     def __init__(self, lma_config: LMAConfigRL, ff_latent_hidden: int, dropout: float, bias: bool):
#         super().__init__()
#         self.ln_1 = LayerNorm(lma_config.d_new, bias=bias)
#         self.attn = LatentAttention_RL(lma_config.d_new, lma_config.num_heads_latent, dropout, bias)
#         self.ln_2 = LayerNorm(lma_config.d_new, bias=bias)
#         self.mlp = LatentMLP_RL(lma_config.d_new, ff_latent_hidden, dropout, bias)

#     def forward(self, z):
#         z = z + self.attn(self.ln_1(z))
#         z = z + self.mlp(self.ln_2(z))
#         return z

# class LMAFeaturesExtractor(BaseFeaturesExtractor):
#     """ Feature extractor using the original LMA mechanism """
#     def __init__(
#         self,
#         observation_space,
#         embed_dim=64, num_heads_stacking=4, target_l_new=3, d_new=32,
#         num_heads_latent=4, ff_latent_hidden=64, num_lma_layers=2,
#         seq_len=6, dropout=0.1, bias=True
#     ):
#         print("\n--- Initializing LMAFeaturesExtractor ---")
#         print("Calculating LMA dimensions...")
#         self.lma_config = LMAConfigRL(
#             seq_len=seq_len, embed_dim=embed_dim, num_heads_stacking=num_heads_stacking,
#             target_l_new=target_l_new, d_new=d_new, num_heads_latent=num_heads_latent
#         )
#         print(f"  Final LMA Config: L={self.lma_config.seq_len}, d0={self.lma_config.embed_dim}, nh_stack={self.lma_config.num_heads_stacking}")
#         print(f"                    L_new={self.lma_config.L_new}, C_new={self.lma_config.C_new}, d_new={self.lma_config.d_new}, nh_latent={self.lma_config.num_heads_latent}")

#         feature_dim = self.lma_config.L_new * self.lma_config.d_new
#         super().__init__(observation_space, features_dim=feature_dim)
#         print(f"  SB3 features_dim (Flattened L_new * d_new): {feature_dim}")

#         self.input_dim_total = observation_space.shape[0]
#         self.seq_len = seq_len
#         if self.input_dim_total % seq_len != 0:
#             raise ValueError(f"Input dimension ({self.input_dim_total}) must be divisible by seq_len ({seq_len}).")
#         self.features_per_step = self.input_dim_total // seq_len

#         self.initial_transform = LMA_InitialTransform_RL(
#             features_per_step=self.features_per_step,
#             lma_config=self.lma_config, dropout=dropout, bias=bias
#         )
#         self.lma_blocks = nn.ModuleList([
#             LMABlock_RL(
#                 lma_config=self.lma_config, ff_latent_hidden=ff_latent_hidden,
#                 dropout=dropout, bias=bias
#             ) for _ in range(num_lma_layers)
#         ])
#         print(f"  Number of LMA Blocks: {num_lma_layers}")
#         self.flatten = nn.Flatten()
#         print("-----------------------------------------")

#     def forward(self, x):
#         batch_size = x.shape[0]
#         try:
#              x_reshaped = x.view(batch_size, self.seq_len, self.features_per_step)
#         except RuntimeError as e:
#              raise RuntimeError(f"Error reshaping input: Input={x.shape}, Target=({batch_size},{self.seq_len},{self.features_per_step})") from e
#         z = self.initial_transform(x_reshaped)
#         for block in self.lma_blocks:
#             z = block(z)
#         features = self.flatten(z)
#         return features

# #===============================================
# # Example Usage
# #===============================================
# if __name__ == "__main__":

#         # Define LMA hyperparameters for testing
#         lma_kwargs = dict(
#             embed_dim=32,           # d0
#             num_heads_stacking=4,   # nh (32 % 4 == 0) -> dk=8
#             target_l_new=3,         # Target L_new (L=6) -> L_new=3 (since 6*32 % 3 == 0)
#             d_new=24,               # d_new
#             num_heads_latent=4,     # Latent heads (24 % 4 == 0) -> latent_dk=6
#             ff_latent_hidden=48,    # Latent MLP hidden (2*d_new)
#             num_lma_layers=2,
#             seq_len=obs_hist_len,
#             dropout=0.1,
#             bias=True
#         )

#         lma_extractor = LMAFeaturesExtractor(
#             observation_space=dummy_obs_space,
#             **lma_kwargs
#         )
