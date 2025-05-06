"""This rewrite is a simplified version of the proposed changes that actually compiles statically in torch 2.0.

This model is the final, optimized crammed model.

Not all ablations discussed in the paper are implemented as switches in this version,
for all those, check scriptable_bert.py on the old branch.

"""
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification

from typing import Optional
from omegaconf import OmegaConf

from .components import (
    _get_norm_fn,
    _get_nonlin_fn,
    EmbeddingComponent,
    PoolingComponent,
    PredictionHeadComponent,
    GLU,
    get_extended_attention_mask,
    _init_module,
)
# Import the specific LMA class if needed for type checking or direct use (optional)
from .attention import get_attention_mechanism
from .latent import InitialLatentTransform, LatentLayer

class crammedBertConfig(PretrainedConfig):
    model_type = "crammedBERT"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)


def construct_crammed_bert(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    config = crammedBertConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    config.arch["embedding"]["vocab_size"] = vocab_size
    config.arch["num_labels"] = downstream_classes

    if downstream_classes is None:
        if config.arch["objective_layout"] == "MLM":
            model = ScriptableLMForPreTraining(config)
        elif config.arch["objective_layout"] == "SCRIPT":
            model = ScriptableLMForSCRIPTTraining(config)
        else:
            raise ValueError(f"Invalid layout {config.arch['objective_layout']} of training objective given.")
    else:
        model = ScriptableLMForSequenceClassification(config)
    return model


class AttentionComponent(torch.nn.Module):
    # No changes needed here, it just wraps whatever get_attention_mechanism returns
    def __init__(self, idx, hidden_size, cfg_attention, use_bias=True):
        super().__init__()
        self.self_attention = get_attention_mechanism(idx, hidden_size, cfg_attention)
        # Check if the instantiated attention module is LMA
        # LMA now handles its own output projection implicitly by returning to H
        # So skip the dense layer if it's LMA
        # ** Correction: The LMABertAttention now returns H dim, so the dense layer
        # ** is still needed IF the output dim isn't H (unlikely for LMA now)
        # ** OR if skip_output_projection is True in config. Let's keep original logic.
        if getattr(cfg_attention, "skip_output_projection", False):
             self.dense = torch.nn.Identity()
        else:
            # Check output dim; LMA now outputs hidden_size
            output_dim = getattr(self.self_attention, "output_dim", hidden_size)
            self.dense = torch.nn.Linear(output_dim, hidden_size, bias=use_bias)


        self.LAYOUT = self.self_attention.LAYOUT

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        # The LMABertAttention module now handles everything internally
        # The dense layer might still be needed if skip_output_projection=False,
        # acting as the standard final projection after attention.
        attn_output = self.self_attention(hidden_states, attention_mask)
        return self.dense(attn_output)


class FFNComponent(torch.nn.Module):
    # No changes needed here, standard FFN
    def __init__(self, hidden_size, intermed_size, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()
        self.dense_in = torch.nn.Linear(hidden_size, intermed_size, bias=use_bias)
        self.nonlin = nonlin_fn()
        if isinstance(self.nonlin, GLU):
            intermed_output_size = intermed_size // 2
        else:
            intermed_output_size = intermed_size
        self.dense_out = torch.nn.Linear(intermed_output_size, hidden_size, bias=use_bias)

    def forward(self, hidden_states):
        return self.dense_out(self.nonlin(self.dense_in(hidden_states)))


class TransformerLayer(torch.nn.Module):
    """
    A transformer-encoder structure.
    MODIFIED: If attention type is LMA, it skips outer Norm/FFN/Residuals.
    """
    def __init__(self, idx, cfg_arch):
        super().__init__()
        self.cfg_arch = cfg_arch # Store config
        self.is_lma = (cfg_arch.attention.type == "lma")

        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)

        # Instantiate Attention Component (could be LMA or standard)
        self.attn = AttentionComponent(
            idx,
            cfg_arch.hidden_size,
            cfg_arch.attention,
            cfg_arch.use_bias,
        )
        self.LAYOUT = self.attn.LAYOUT # Inherit layout

        # Only instantiate outer Norm/FFN if NOT LMA
        if not self.is_lma:
            self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
            self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
            self.ffn = FFNComponent(
                cfg_arch.hidden_size,
                cfg_arch.intermed_size,
                _get_nonlin_fn(cfg_arch.nonlin),
                cfg_arch.use_bias,
            )
        else:
            # Create dummy modules or None if LMA, as they are bypassed
            self.norm1 = torch.nn.Identity() # Not used
            self.norm2 = torch.nn.Identity() # Not used
            self.ffn = torch.nn.Identity()   # Not used

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):
        if self.is_lma:
            # --- LMA Path ---
            # The refactored LMABertAttention handles its internal norms,
            # FFN, and residuals. Just call it directly.
            # The outer dropout is applied to the final output of the LMA block.
            states = self.dropout(self.attn(states, attention_mask))
            # NOTE: The original paper structure implies dropout within the
            # residual adds inside LMA. Applying it outside here might differ slightly.
            # Consider moving dropout inside LMABertAttention's residual adds if needed.
        else:
            # --- Standard Pre-LN Path ---
            normed_states_for_attn = self.norm1(states)
            attn_output = self.attn(normed_states_for_attn, attention_mask)
            states = states + self.dropout(attn_output) # Residual connection 1

            normed_states_for_ffn = self.norm2(states)
            ffn_output = self.ffn(normed_states_for_ffn)
            states = states + self.dropout(ffn_output) # Residual connection 2

        return states

# --- ScriptableLM and downstream heads remain unchanged ---
# They interact with the output of the TransformerLayer stack,
# which now correctly returns [B, S, H] for both LMA and standard attention.

class ScriptableLM(PreTrainedModel):
    """Simplified transformer wrapper."""
    # (Content remains the same as your provided file)
    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.embedding = EmbeddingComponent(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)

        # --- NEW latent front‑end ---
        self.latent_front = InitialLatentTransform(self.cfg.hidden_size, self.cfg.attention)

        # Build latent Transformer stack
        self.layers = torch.nn.ModuleList([
            LatentLayer(
                d_new=self.cfg.attention.d_new,
                nh_latent=self.cfg.attention.num_heads_latent,
                ff_hidden=self.cfg.attention.ff_latent_hidden,
                dropout=self.cfg.hidden_dropout_prob,
                bias=self.cfg.use_bias,
            )
            for _ in range(self.cfg.num_transformer_layers)
        ])      
        
        self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            latent_dim = int(self.cfg.attention.d_new)
            self.final_norm = _get_norm_fn(self.cfg.norm)(latent_dim, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)                      # [B,S,H]

        if attention_mask is not None:                    # [B,S] → float 0/1
            if attention_mask.dim()==2:  attention_mask = attention_mask.float()
            if attention_mask.dim()==4:  attention_mask = (attention_mask>-0.5).float().squeeze(1).squeeze(1)

        z, latent_mask = self.latent_front(x, attention_mask)    # -> latent space

        for blk in self.layers:
            z = blk(z, latent_mask)

        z = self.final_norm(z)                              # still on d_new
        return z                                            # [B,L_new,d_new]


class ScriptableLMForPreTraining(PreTrainedModel):
    # (Content remains the same as your provided file)
    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableLM(config)

        # project latent d_new -> original hidden/embedding dim (H)
        # (No longer used; kept for compatibility)
        self.latent_to_hidden = torch.nn.Linear(
            self.cfg.attention.d_new,
            self.cfg.embedding.embedding_dim,
            bias=self.cfg.use_bias,
        )

        if not self.cfg.skip_head_transform:
            self.prediction_head = PredictionHeadComponent(self.cfg)
        else:
            self.prediction_head = torch.nn.Identity()  # from linear in old version

        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        # Tie weights if configured
        if self.cfg.tie_weights:
            self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sparse_prediction = self.cfg.sparse_prediction

        self._init_weights()

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )
        # Tie weights explicitly after initialization if needed (redundant if done in init)
        if self.cfg.tie_weights:
             self.decoder.weight = self.encoder.embedding.word_embedding.weight


    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        # latent outputs  [B , L_new , d_new]
        z_latent = self.encoder(input_ids, attention_mask)

        # Convert back to token‑level hidden states [B,S,H]
        h_tokens = self.encoder.latent_front.inverse_transform(z_latent)

        logits = self.decoder(self.prediction_head(h_tokens))  # [B,S,vocab]

        # -------- Loss ---------
        if labels is not None:
            if self.sparse_prediction:
                loss = self._forward_sparse(h_tokens, labels)
            else:
                loss = self.loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
        else:
            loss = logits.new_zeros((1,))

        return {"loss": loss, "logits": logits}


    def _forward_sparse(self, hidden_states: torch.Tensor, labels: Optional[torch.Tensor] = None):
        labels = labels.view(-1)
        mask_positions = labels != self.loss_fn.ignore_index
        num_masks_guaranteed = round(self.sparse_prediction * labels.numel())
        indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]

        hidden_states = hidden_states.view(-1, hidden_states.size(-1))[indices]  # sparse hidden
        labels = labels[indices]
        processed_outputs = self.prediction_head(hidden_states) # Apply head to sparse outputs
        logits = self.decoder(processed_outputs)
        masked_lm_loss = self.loss_fn(logits, labels)
        return masked_lm_loss


class ScriptableLMForSequenceClassification(PreTrainedModel):
    """Classification head and pooler."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.num_labels = self.cfg.num_labels

        self.encoder = ScriptableLM(config)
        # REMOVE or ignore self.latent_to_hidden - no longer needed here
        # self.latent_to_hidden = torch.nn.Linear(...) # <<< DELETE / IGNORE

        self.pooler = PoolingComponent(self.cfg.classification_head, self.cfg.hidden_size) # Pooler operates on H
        # The head dimension should match the pooler's output dimension, which is derived from H
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
        self._init_weights() # Ensure pooler/head are initialized

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
             # Initialize pooler and head layers as well if not covered by _init_module specific checks
             if isinstance(module, (torch.nn.Linear)) and module is not self.encoder.latent_front.from_latent and module is not self.encoder.latent_front.to_latent : # Example condition
                 _init_module(
                     module,
                     self.cfg.init.type,
                     self.cfg.init.std,
                     self.cfg.hidden_size, # Use H for standard layers
                     self.cfg.num_transformer_layers,
                 )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        # Get final latent output from the encoder
        z_latent = self.encoder(input_ids, attention_mask)             # [B, L_new, d_new]

        # --- FIX: Inverse transform back to token space BEFORE pooling ---
        # Use the inverse_transform method from the encoder's latent_front
        h_tokens = self.encoder.latent_front.inverse_transform(z_latent) # [B, S, H]
        # -----------------------------------------------------------------

        # Now, pool based on the reconstructed token sequence
        pooled_output = self.pooler(h_tokens)                          # Pooler operates on [B, S, H]
                                                                       # likely extracts h_tokens[:, 0, :]

        # The rest should work as intended now
        logits = self.head(pooled_output)

        # --- Loss Calculation (remains the same, operates on final logits) ---
        if labels is not None:
            if self.problem_type is None:
                if self.num_labels == 1: self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int): self.problem_type = "single_label_classification"
                else: self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.squeeze() if self.num_labels==1 else logits, labels.squeeze() if self.num_labels==1 else labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        else:
            loss = logits.new_zeros((1,))

        return dict(logits=logits, loss=loss)


class ScriptableLMForSCRIPTTraining(PreTrainedModel):
    """Pretraining machinery using SCRIPT. Needs inverse transform."""

    config_class = crammedBertConfig
    ALPHA = 1.0  # SCRIPT constant

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        # SCRIPT doesn't use num_labels from config in the same way?
        # It uses vocab size for MLM-like generation.

        self.encoder = ScriptableLM(config)
        # Prediction head should operate on H dimension
        self.prediction_head = PredictionHeadComponent(self.cfg)

        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        if self.cfg.tie_weights:
            self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sparse_prediction = self.cfg.sparse_prediction
        assert self.sparse_prediction # SCRIPT relies on sparse

        self._init_weights()

    def _init_weights(self, module=None):
        # ...(Initialization logic remains the same)...
        modules = self.modules() if module is None else [module]
        for module in modules:
             if isinstance(module, torch.nn.Linear) and module is not self.encoder.latent_front.from_latent and module is not self.encoder.latent_front.to_latent : # Example condition
                 _init_module(
                     module,
                     self.cfg.init.type,
                     self.cfg.init.std,
                     self.cfg.hidden_size, # Use H for standard layers
                     self.cfg.num_transformer_layers,
                 )
        if self.cfg.tie_weights:
             self.decoder.weight = self.encoder.embedding.word_embedding.weight


    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        loss = torch.tensor(0.0, dtype=torch.float, device=input_ids.device)

        # Get original sequence length S
        B, S = input_ids.shape

        # --- Encoder Pass ---
        # latent outputs  [B, L_new, d_new]
        z_latent = self.encoder(input_ids, attention_mask)
        # Convert back to token‑level hidden states [B, S, H]
        h_tokens = self.encoder.latent_front.inverse_transform(z_latent)

        # Reshape for processing
        outputs_flat = h_tokens.view(-1, h_tokens.size(-1)) # [B*S, H]

        if labels is not None:
            # ## Generation pass (MLM part of SCRIPT) ##
            labels_flat = labels.view(-1)
            mask_positions = labels_flat != self.loss_fn.ignore_index
            num_masks_guaranteed = round(self.sparse_prediction * labels_flat.numel())
            indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]

            # Select sparse outputs using indices from the original sequence dimension
            outputs_sparse = outputs_flat[indices] # [N_masked, H]
            labels_sparse = labels_flat[indices]   # [N_masked]

            # Apply prediction head and decoder to sparse H representations
            logits_gen = self.decoder(self.prediction_head(outputs_sparse)) # [N_masked, V]
            loss += self.loss_fn(logits_gen, labels_sparse) # Generation loss

            # ## Discrimination pass (ELECTRA part of SCRIPT) ##
            with torch.no_grad(): # Detach generator logits for sampling
                 resampled_token_ids = self._gumbel_sample(logits_gen) # [N_masked]

            # Create discriminator input by replacing original masked tokens
            discriminator_input_ids_flat = input_ids.clone().view(-1)
            discriminator_input_ids_flat[indices] = resampled_token_ids
            discriminator_input_ids = discriminator_input_ids_flat.view(B, S)

            # Create critic labels based on whether token was replaced
            critic_labels_flat = (input_ids.view(-1) != discriminator_input_ids_flat).to(h_tokens.dtype) # [B*S]

            # --- Run Encoder AGAIN on corrupted input ---
            z_latent_disc = self.encoder(discriminator_input_ids, attention_mask)
            h_tokens_disc = self.encoder.latent_front.inverse_transform(z_latent_disc) # [B, S, H]
            outputs_disc_flat = h_tokens_disc.view(-1, h_tokens_disc.size(-1)) # [B*S, H]

            # --- Get discriminator logits ---
            # Apply prediction head and decoder to ALL H representations from corrupted input
            disc_logits_full = self.decoder(self.prediction_head(outputs_disc_flat)) # [B*S, V]
            # Convert to binary logits (replaced vs. original)
            binary_logits = self._get_binary_logits(disc_logits_full) # [B*S]

            # --- Calculate Discriminator Loss ---
            # Apply loss ONLY at original token positions (B*S dimension)
            loss += self.ALPHA * torch.nn.functional.binary_cross_entropy_with_logits(
                binary_logits, # [B*S]
                critic_labels_flat # [B*S]
            )

        else:
            # If no labels, just calculate logits for potential inference/eval?
            # Need to decide what the output should be. Maybe just return 0 loss?
            logits = self.decoder(self.prediction_head(outputs_flat.view_as(h_tokens))) # Calculate full logits
            loss += logits.new_zeros((1,))


        # Decide what logits to return - generator logits? discriminator logits?
        # Returning generator logits (sparse) might be confusing.
        # Returning full logits from first pass might be better for consistency?
        logits_to_return = self.decoder(self.prediction_head(outputs_flat.view_as(h_tokens)))

        return {"loss": loss, "logits": logits_to_return} # Return full logits from 1st pass

    # _get_binary_logits, _gumbel_sample, _gumbel_noise remain the same
    def _get_binary_logits(self, logits):
        return torch.logsumexp(logits, dim=-1)

    def _gumbel_sample(self, logits, temperature=1.0):
        return ((logits / temperature) + self._gumbel_noise(logits)).argmax(dim=-1)

    def _gumbel_noise(self, inputs, eps=1e-9):
        noise = torch.zeros_like(inputs).uniform_(0, 1)
        return -torch.log(-torch.log(noise + eps) + eps)


class ScriptableLMForTokenClassification(PreTrainedModel):
    """Classification head without pooling."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.num_labels = config.num_labels # Get num_labels from config directly

        self.encoder = ScriptableLM(config)
        # REMOVE or ignore self.latent_to_hidden - no longer needed here
        # self.latent_to_hidden = torch.nn.Linear(...) # <<< DELETE / IGNORE

        # The final head operates on the hidden dimension H
        # Assuming classification_head.head_dim is H? If not, adjust.
        # If prediction_head component is used before this, ensure it outputs H.
        # Let's assume head takes H directly for simplicity now.
        self.head = torch.nn.Linear(self.cfg.hidden_size, self.num_labels) # Input is H

        self.problem_type = None
        self._init_weights() # Ensure head is initialized

    def _init_weights(self, module=None):
        # ...(Initialization logic remains the same)...
        modules = self.modules() if module is None else [module]
        for module in modules:
             if isinstance(module, torch.nn.Linear) and module is not self.encoder.latent_front.from_latent and module is not self.encoder.latent_front.to_latent : # Example condition
                _init_module(
                    module,
                    self.cfg.init.type,
                    self.cfg.init.std,
                    self.cfg.hidden_size, # Use H for standard layers
                    self.cfg.num_transformer_layers,
                )


    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        # latent outputs  [B , L_new , d_new]
        z_latent = self.encoder(input_ids, attention_mask)

        # ----> APPLY INVERSE TRANSFORM HERE <----
        # Convert back to token‑level hidden states [B, S, H]
        h_tokens = self.encoder.latent_front.inverse_transform(z_latent)

        # Apply head to the reconstructed h_tokens
        logits = self.head(h_tokens) # Shape: [B, S, num_labels]

        # --- Loss Calculation (remains the same, operates on final logits) ---
        if labels is not None:
             # ...(problem type detection and loss calculation remains the same)...
             # Loss needs logits [B*S, num_labels] and labels [B*S]
             if self.problem_type is None:
                 # ...(problem type detection)...
                 if self.num_labels == 1: self.problem_type = "regression" # Unlikely for token class.
                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int): self.problem_type = "single_label_classification"
                 else: self.problem_type = "multi_label_classification" # Possible if multi-label per token

             if self.problem_type == "regression":
                 loss_fct = torch.nn.MSELoss()
                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)) # Adjust shapes if needed
             elif self.problem_type == "single_label_classification":
                 loss_fct = torch.nn.CrossEntropyLoss()
                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
             elif self.problem_type == "multi_label_classification":
                 loss_fct = torch.nn.BCEWithLogitsLoss()
                 loss = loss_fct(logits, labels) # Assumes labels have shape [B, S, num_labels]
        else:
            loss = logits.new_zeros((1,))

        return dict(logits=logits, loss=loss)


# ###### HF registry here ############### #

AutoConfig.register("crammedBERT", crammedBertConfig)
AutoModel.register(crammedBertConfig, ScriptableLM)
AutoModelForMaskedLM.register(crammedBertConfig, ScriptableLMForPreTraining)
AutoModelForSequenceClassification.register(crammedBertConfig, ScriptableLMForSequenceClassification)
AutoModelForTokenClassification.register(crammedBertConfig, ScriptableLMForTokenClassification)
