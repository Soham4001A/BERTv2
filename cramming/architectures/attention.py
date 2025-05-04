"""Attention modules. The final model uses "self-attention", but other options were tried and are still documented here."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertSelfAttention
from .embeddings import Rotary, RotarySanityCheck, RotaryEleutherAI, RotaryLLAMA
from typing import Optional
from einops.layers.torch import Rearrange
from einops import rearrange
import math # Make sure math is imported

# ────────────────────────────────────────────────────────────────────────────
# Latent Meta Attention – NLP variant
# Ports the RL implementation to '[B,S,H]' tensors used in BERT.
# ────────────────────────────────────────────────────────────────────────────
import types

def _find_closest_divisor(n: int, target: int) -> int:
    """
    Find a divisor of n that is as close as possible to 'target'.
    Guarantees the return value divides n.
    """
    if target <= 0: target = 1 # Ensure target is positive
    if n == 0: return 1 # Avoid division by zero if total features is 0 somehow

    # Check if target itself is a divisor
    if n % target == 0:
        return target

    # Search outwards from the target
    low = target - 1
    high = target + 1
    while low > 0 or high <= n:
        if low > 0 and n % low == 0:
            return low
        if high <= n and n % high == 0:
            return high
        low -= 1
        high += 1
    return 1 # Fallback, should only happen if n=1

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

# ---- Latent‑space primitives (used by LMABertAttention internally) -------
class _LatentAttentionInternal(torch.nn.Module):
    """
    INTERNAL LMA component: Performs Multi‑head attention within latent space.
    Takes Z (output of to_latent) as input.
    Returns the attention output projection *without* residual connection.
    """
    def __init__(self, d_new: int, num_heads: int, dropout: float, bias: bool):
        super().__init__()
        assert d_new % num_heads == 0, "d_new must be divisible by num_heads"
        self.d_new   = d_new
        self.nh      = num_heads
        self.dk      = d_new // num_heads
        self.flash   = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        # --- Layers ---
        self.c_attn  = torch.nn.Linear(d_new, 3 * d_new, bias=bias) # Input proj to QKV
        self.c_proj  = torch.nn.Linear(d_new, d_new, bias=bias) # Output projection
        self.at_drop = torch.nn.Dropout(dropout)
        # NOTE: Removed self.res_drop, dropout is handled in the main block now

    def forward(self, z_normed, mask: Optional[torch.Tensor] = None):
        # Input z_normed is ALREADY LayerNormed
        B, L, _ = z_normed.shape
        q, k, v = self.c_attn(z_normed).chunk(3, dim=-1)            # [B,L,d_new]×3
        q = q.view(B, L, self.nh, self.dk).transpose(1, 2)          # [B,nh,L,dk]
        k = k.view(B, L, self.nh, self.dk).transpose(1, 2)
        v = v.view(B, L, self.nh, self.dk).transpose(1, 2)

        # ----- Attention computation -----
        # Correction: Use SDPA with key_padding_mask when mask is available
        if self.flash:
            # Note: SDPA needs boolean mask where True indicates padding/masking
            attn_mask_bool = mask if mask is not None else None
            y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None, # Not needed for standard padding
                    dropout_p=self.at_drop.p if self.training else 0.0,
                    is_causal=False,
                    # Use key_padding_mask
                    key_padding_mask=attn_mask_bool
                    )
        else: # Manual path
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dk))
            if mask is not None: # mask shape [B,L] boolean; True = pad
                att = att.masked_fill(mask[:, None, None, :], float("-inf"))  # mask TO padded keys
                att = att.masked_fill(mask[:, None, :, None], float("-inf"))  # mask FROM padded queries
            att = torch.nn.functional.softmax(att, dim=-1)
            att = self.at_drop(att) # Apply dropout AFTER softmax
            y   = att @ v

        # Reassemble heads and apply output projection
        y = y.transpose(1, 2).reshape(B, L, self.d_new) # [B,L,d_new]
        attn_output = self.c_proj(y) # Apply final projection
        return attn_output # Return projected attention output, NO residual/dropout here

class _LatentMLPInternal(torch.nn.Module):
    """
    INTERNAL LMA component: Feed‑forward network in latent space.
    Takes output of first residual+LN as input.
    Returns MLP output *without* residual connection.
    """
    def __init__(self, d_new: int, hidden: int, dropout: float, bias: bool):
        super().__init__()
        # --- Layers ---
        self.c_fc    = nn.Linear(d_new, hidden, bias=bias)
        self.act     = nn.GELU() # Or other activation if needed
        self.dropout1= nn.Dropout(dropout) # Dropout after activation
        self.c_proj  = nn.Linear(hidden, d_new, bias=bias)
        # NOTE: Removed final dropout, handled in the main block now

    def forward(self, x_normed):
        # Input x_normed is ALREADY LayerNormed
        x = self.c_fc(x_normed)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.c_proj(x)
        return x # Return MLP output, NO residual/dropout here

# ---- Main LMA module implementing the full paper logic ----
class LMABertAttention(torch.nn.Module):
    """
    Latent Meta Attention (LMA) aligned with paper description (Section 2.4).
    Performs sequence transformation, latent attention, latent FFN,
    and residual connections *within the latent space*, before transforming back.

    Replaces the standard AttentionComponent + FFNComponent logic when used
    in the modified TransformerLayer.
    """
    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]" # Still consumes/produces this shape for compatibility layer

    def __init__(self, hidden_size: int, cfg_attention):
        super().__init__()
        # hyper‑parameters from cfg_attention
        self.hidden_size        = hidden_size
        self.nh_stack           = int(getattr(cfg_attention, "num_heads_stacking", 4))
        self.d_new              = int(getattr(cfg_attention, "d_new", hidden_size // 2))
        self.nh_latent          = int(getattr(cfg_attention, "num_heads_latent", 4))
        self.ff_latent_hidden   = int(getattr(cfg_attention, "ff_latent_hidden", 4 * self.d_new))
        self.n_latent_blocks_ignored = int(getattr(cfg_attention, "num_blocks", 1)) # Renamed, logic moved here
        if self.n_latent_blocks_ignored > 1:
             print("WARNING: LMABertAttention currently implements only 1 effective latent block per layer for simplicity. num_blocks > 1 ignored.")
        self.target_l_new_cfg   = getattr(cfg_attention, "target_l_new", None)
        self.dropout_prob       = cfg_attention.dropout_prob
        self.bias               = cfg_attention.qkv_bias

        if hidden_size % self.nh_stack != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by num_heads_stacking {self.nh_stack}")

        # Build parameters lazily or statically
        self._built = False
        static_seq_len = getattr(cfg_attention, "static_seq_len", None)
        if static_seq_len is not None:
            self._build(static_seq_len, torch.device("cpu"))

        # This module outputs the original hidden_size
        self.output_dim = hidden_size

    # ---------------------------------------------------------------------
    def _build(self, seq_len: int, device: torch.device):
        """Create sub‑modules for a fixed sequence length."""
        # --- Calculate derived dimensions ---
        self.dk_stack   = self.hidden_size // self.nh_stack # Dim per head after stacking
        total_features  = seq_len * self.hidden_size
        target_l_new    = (self.target_l_new_cfg
                           if self.target_l_new_cfg is not None
                           else max(2, seq_len // 2))
        self.L_new      = _find_closest_divisor(total_features, target_l_new)
        self.C_new      = total_features // self.L_new

        # --- Layers ---
        # Projection into latent space
        self.to_latent  = torch.nn.Linear(self.C_new, self.d_new, bias=self.bias)
        # LayerNorms within latent space
        self.ln1        = _LayerNorm(self.d_new, bias=self.bias)
        self.ln2        = _LayerNorm(self.d_new, bias=self.bias)
        # Latent Attention (Internal - no residual)
        self.latent_attn= _LatentAttentionInternal(self.d_new, self.nh_latent, self.dropout_prob, self.bias)
        # Latent MLP (Internal - no residual)
        self.latent_mlp = _LatentMLPInternal(self.d_new, self.ff_latent_hidden, self.dropout_prob, self.bias)
        # Dropouts for residual connections
        self.res_drop1  = torch.nn.Dropout(self.dropout_prob)
        self.res_drop2  = torch.nn.Dropout(self.dropout_prob)
        # Projection back from latent space
        self.from_latent= torch.nn.Linear(self.d_new, self.C_new, bias=self.bias)

        # Move layers to the correct device
        self.to_latent   = self.to_latent.to(device)
        self.ln1         = self.ln1.to(device)
        self.ln2         = self.ln2.to(device)
        self.latent_attn = self.latent_attn.to(device)
        self.latent_mlp  = self.latent_mlp.to(device)
        self.res_drop1   = self.res_drop1.to(device)
        self.res_drop2   = self.res_drop2.to(device)
        self.from_latent = self.from_latent.to(device)

        # Cache constants
        self.seq_len = seq_len
        self.register_buffer("_dummy", torch.empty(0, device=device)) # Track device
        self._built  = True

    # ---------------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None):
        """
        Implements the full LMA block logic internally, including latent residuals.
        """
        B, S, H = hidden_states.shape

        # --- 1. Build if needed ---
        if not self._built:
            self._build(S, hidden_states.device)
        elif S != self.seq_len:
            raise RuntimeError(f"LMABertAttention static build mismatch: Expected S={self.seq_len}, got S={S}")

        # --- 2. Initial Mask Processing & Input Zeroing ---
        input_mask = None # This is the original [B,S,1] float mask for final output zeroing
        if attention_mask is not None:
            if attention_mask.dim() == 4: # [B,1,1,S] HF format
                input_mask = (attention_mask > -0.5).float().squeeze(1).squeeze(1) # [B,S]
            else: # [B,S] or [B,1,S]
                input_mask = attention_mask.float().view(B, S)
            input_mask = input_mask.unsqueeze(-1) # [B,S,1]
            hidden_states = hidden_states * input_mask # Zero out padding inputs

        # --- 3. LMA Stage 2a: Head Stacking ---
        head_views = torch.split(hidden_states, self.dk_stack, dim=2)
        x_stacked  = torch.cat(head_views, dim=1) # [B, S*nh_stack, dk_stack]
        x_stacked  = x_stacked * (1.0 / math.sqrt(self.nh_stack)) # Keep variance ~constant

        # --- 4. LMA Stage 2b: Re-Chunking & Project to Latent (Z) ---
        flat_stacked = x_stacked.view(B, -1)             # [B, S*H]
        x_chunks     = flat_stacked.view(B, self.L_new, self.C_new) # [B, L_new, C_new]
        z            = self.to_latent(x_chunks)          # [B, L_new, d_new] -> This is LATENT INPUT Z

        # --- 5. Derive Latent Mask ---
        latent_mask = None
        if input_mask is not None:
            # Corrected robust way to derive boolean latent mask (True = PAD)
            # Checks if the input chunks to to_latent were all zeros
            latent_mask = (torch.abs(x_chunks).sum(dim=-1) < 1e-9) # [B, L_new] boolean

        # --- 6. Latent Space Processing (Paper Eq. 10 & 12 style) ---
        # Residual Path 1: Z + Attn(LN1(Z))
        z_norm1     = self.ln1(z)
        attn_output = self.latent_attn(z_norm1, latent_mask)
        z_res1      = z + self.res_drop1(attn_output) # First latent residual connection

        # Residual Path 2: Z_res1 + MLP(LN2(Z_res1))
        z_norm2     = self.ln2(z_res1)
        mlp_output  = self.latent_mlp(z_norm2)
        z_out_latent= z_res1 + self.res_drop2(mlp_output) # Second latent residual connection

        # --- 7. Project Back & Inverse Reshape ---
        chunks_back = self.from_latent(z_out_latent)          # [B, L_new, C_new]
        flat_back   = chunks_back.reshape(B, -1)              # [B, S*H]
        x_stacked_b = flat_back.view(B, S * self.nh_stack, self.dk_stack) # [B, S*nh_stack, dk_stack]

        # Inverse head stacking
        x_unstacked = x_stacked_b.view(B, S, self.nh_stack, self.dk_stack)
        out         = torch.cat(torch.unbind(x_unstacked, dim=2), dim=2) # [B, S, H]

        # --- 8. Final Output Zeroing ---
        if input_mask is not None:
            out = out * input_mask # Ensure padded outputs are zero

        return out # Return tensor in the original [B,S,H] shape

# --- Rest of the standard attention mechanisms remain unchanged ---
# (SeqFirstSelfAttention, PyTorch wrappers, FlashAttention, etc.)

def get_attention_mechanism(
    idx,
    hidden_size,
    cfg_attention,
):
    # This function remains largely the same, but LMABertAttention is now different
    if cfg_attention.type == "self-attention":
        mechanism = SeqFirstSelfAttention(hidden_size, cfg_attention)
    # ... other attention types ...
    elif cfg_attention.type == "lma":
        # Now instantiates the refactored LMA module
        mechanism = LMABertAttention(hidden_size, cfg_attention)
    else:
        raise ValueError(f"Invalid attention type {cfg_attention.type} given.")
    return mechanism


# def get_attention_mechanism(
#     idx,
#     hidden_size,
#     cfg_attention,
# ):
#     if cfg_attention.type == "self-attention":
#         mechanism = SeqFirstSelfAttention(hidden_size, cfg_attention)  # neox
#     elif cfg_attention.type == "pytorch":
#         # Sanity check 1: [Warning: This includes the output projection twice...]
#         mechanism = SelfAttentionPyTorch(hidden_size, cfg_attention)  # torch default
#     elif cfg_attention.type == "pytorch-seqfirst":
#         # Sanity check 1: [Warning: This includes the output projection twice...]
#         mechanism = SeqFirstSelfAttentionPyTorch(hidden_size, cfg_attention)  # torch default
#     elif cfg_attention.type == "huggingface":
#         mechanism = BertAttentionWrapper(hidden_size, cfg_attention)  # always includes bias!
#     elif cfg_attention.type == "flash-attention-impl":  # the fast implementation called flash
#         mechanism = FlashMultiHeadAttention(hidden_size, cfg_attention)
#     elif cfg_attention.type == "fourier":
#         mechanism = FourierMixing(hidden_size, cfg_attention)
#     elif cfg_attention.type == "fourier-experimental":
#         mechanism = FourierMixingParametrized(hidden_size, cfg_attention)
#     elif cfg_attention.type == "flash":  # flash from transformer quality in linear time
#         mechanism = FLASH(hidden_size, cfg_attention)
#     elif cfg_attention.type == "tuformer":
#         mechanism = TuFormAttention(hidden_size, cfg_attention)
#     elif cfg_attention.type == "funnel":  # dont use this with a normal seq->seq model
#         mechanism = FunnelAttention(hidden_size, cfg_attention)
#     elif cfg_attention.type == "seqfirst_tuformer":
#         mechanism = SeqFirstTuFormAttention(hidden_size, cfg_attention)
#     elif cfg_attention.type == "seqfirst2_tuformer":
#         mechanism = SeqFirstTuFormAttention(hidden_size, cfg_attention)
#     elif cfg_attention.type == "none":
#         mechanism = Identity(hidden_size)
#     elif cfg_attention.type == "fourier-hybrid":
#         if idx in cfg_attention.hybrid_layers:
#             mechanism = SeqFirstSelfAttention(hidden_size, cfg_attention)
#         else:
#             mechanism = FourierMixing(hidden_size, cfg_attention)
#     elif cfg_attention.type == "lma":
#         mechanism = LMABertAttention(hidden_size, cfg_attention)
#     else:
#         raise ValueError(f"Invalid attention type {cfg_attention.type} given.")
#     return mechanism


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
