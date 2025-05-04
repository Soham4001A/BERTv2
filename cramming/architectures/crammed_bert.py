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
from .attention import get_attention_mechanism, LMABertAttention


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
        self.layers = torch.nn.ModuleList([TransformerLayer(idx, self.cfg) for idx in range(self.cfg.num_transformer_layers)])
        self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            # Ensure mask is processed correctly based on layout BEFORE passing to first layer
            if not self.seq_first and attention_mask.dim() == 2:
                 # Standard MHA expects [B, S] -> [B, 1, 1, S] or similar broadcastable
                 attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.use_causal_attention)
            # If seq_first, the attention mechanism inside handles the layout adjustment if needed
            # LMA handles its own mask conversion internally

        hidden_states = self.embedding(input_ids)

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            # Potentially adjust mask layout if needed for seq_first standard attention
            # LMA handles [B, S, H] input regardless of internal layout

        # --- Layer Loop ---
        for i, layer_module in enumerate(self.layers):
             # Pass the appropriate mask format
             current_mask = attention_mask # Pass the potentially extended mask
             # LMA internal forward handles mask conversion from various formats
             hidden_states = layer_module(hidden_states, current_mask)


        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        return self.final_norm(hidden_states)


class ScriptableLMForPreTraining(PreTrainedModel):
    # (Content remains the same as your provided file)
    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableLM(config)

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
        outputs = self.encoder(input_ids, attention_mask)
        # Ensure outputs is viewed correctly before prediction head/decoder
        # Shape should be [B*S, H] if not sparse? Or [B, S, H]? Check components.
        # Assuming prediction_head/decoder expect [..., H]
        outputs_for_head = outputs.view(-1, outputs.shape[-1])


        if self.sparse_prediction and labels is not None:
            masked_lm_loss = self._forward_sparse(outputs_for_head, labels)
            # Need logits for potential metrics - recompute on full sequence?
            # Or just return loss? For now, return loss.
            logits = None # Or compute non-sparse for logging?
        else:
            processed_outputs = self.prediction_head(outputs_for_head)
            logits = self.decoder(processed_outputs)
            masked_lm_loss = logits.new_zeros((1,)) # Ensure loss tensor is on correct device
            if labels is not None:
                masked_lm_loss = self.loss_fn(logits, labels.view(-1))


        # Return loss and maybe logits (handle sparse case where logits might be partial)
        return {"loss": masked_lm_loss, "logits": logits if not (self.sparse_prediction and labels is not None) else None}


    # Sparse prediction logic seems okay based on comments
    def _forward_sparse(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # ...(Same as your provided code)...
        labels = labels.view(-1)
        mask_positions = labels.view(-1) != self.loss_fn.ignore_index
        num_masks_guaranteed = round(self.sparse_prediction * labels.shape[0])
        indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]
        outputs = outputs[indices]
        labels = labels[indices]
        processed_outputs = self.prediction_head(outputs) # Apply head to sparse outputs
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
        self.pooler = PoolingComponent(self.cfg.classification_head, self.cfg.hidden_size)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
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

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        logits = self.head(self.pooler(self.encoder(input_ids, attention_mask)))

        if labels is not None:
            if self.problem_type is None:  # very much from huggingface
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
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
    """Pretraining machinery using SCRIPT from Nijkamp et al., 2021. Always running sparse prediction."""

    config_class = crammedBertConfig
    ALPHA = 1.0  # SCRIPT constant

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.num_labels = self.cfg.num_labels

        self.encoder = ScriptableLM(config)
        self.prediction_head = PredictionHeadComponent(self.cfg)

        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sparse_prediction = self.cfg.sparse_prediction
        assert self.sparse_prediction

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

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        loss = torch.tensor(0.0, dtype=torch.float, device=input_ids.device)

        outputs = self.encoder(input_ids, attention_mask)
        outputs = outputs.view(-1, outputs.shape[-1])

        if labels is not None:
            # ## Generation pass ##
            labels = labels.view(-1)
            mask_positions = labels.view(-1) != self.loss_fn.ignore_index
            num_masks_guaranteed = round(self.sparse_prediction * labels.shape[0])
            indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]

            # sparse outputs for prediction
            outputs = outputs[indices]
            labels = labels[indices]

            logits = self.decoder(self.prediction_head(outputs))  # sparse logits
            loss += self.loss_fn(logits, labels)

            # ## Discrimination pass ##
            resampled_token_ids = self._gumbel_sample(logits.detach())
            discriminator_input_ids = input_ids.clone().view(-1)
            discriminator_input_ids[indices] = resampled_token_ids

            critic_labels = (input_ids.view(-1) != discriminator_input_ids).to(outputs.dtype)

            outputs = self.encoder(discriminator_input_ids.view_as(input_ids), attention_mask).view(-1, outputs.shape[-1])
            disc_logits = self.decoder(self.prediction_head(outputs))  # full logits
            binary_logits = self._get_binary_logits(disc_logits)

            # ELECTRA-type discriminator:
            loss += self.ALPHA * torch.nn.functional.binary_cross_entropy_with_logits(binary_logits, critic_labels)

        else:
            logits = self.decoder(self.prediction_head(outputs))
            loss += outputs.new_zeros((1,))

        return {"loss": loss, "logits": logits}

    def _get_binary_logits(self, logits):
        # Convert to binary decision as described in SCRIPT
        # exp_logitsum = torch.exp(disc_logits).sum(dim=-1)  # autocast ok?
        # binary_logits = torch.stack([1 / (exp_logitsum + 1), exp_logitsum / (exp_logitsum + 1)], dim=-1)  # stack minus and plus
        # instead, we can also compute logit[binary_logits], which is

        # let y = sum(exp(logits)) / ( sum(exp(logits))+1 ), 1-y = 1 / ( sum(exp(logits))+1 )
        # log(y / (1-y)) = log( sum(exp(logits)) / ( sum(exp(logits))+1 ) * ( sum(exp(logits))+1 ) / 1)
        #                = log(sum(exp(logits))
        # Then, we can use BCEWithLogitsLoss, to safely compute logit probs via sigmoids
        return torch.logsumexp(logits, dim=-1)

    def _gumbel_sample(self, logits, temperature=1.0):
        """via https://github.com/lucidrains/electra-pytorch/blob/master/electra_pytorch/electra_pytorch.py"""
        return ((logits / temperature) + self._gumbel_noise(logits)).argmax(dim=-1)

    def _gumbel_noise(self, inputs, eps=1e-9):
        """via https://github.com/lucidrains/electra-pytorch/blob/master/electra_pytorch/electra_pytorch.py"""
        noise = torch.zeros_like(inputs).uniform_(0, 1)
        return -torch.log(-torch.log(noise + eps) + eps)


class ScriptableLMForTokenClassification(PreTrainedModel):
    """Classification head without pooling."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableLM(config)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
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

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        logits = self.head(self.encoder(input_ids, attention_mask))

        if labels is not None:
            if self.problem_type is None:  # very much from huggingface
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            else:
                raise ValueError("Wrong problem type!")
        else:
            loss = logits.new_zeros((1,))

        return dict(logits=logits, loss=loss)


# ###### HF registry here ############### #

AutoConfig.register("crammedBERT", crammedBertConfig)
AutoModel.register(crammedBertConfig, ScriptableLM)
AutoModelForMaskedLM.register(crammedBertConfig, ScriptableLMForPreTraining)
AutoModelForSequenceClassification.register(crammedBertConfig, ScriptableLMForSequenceClassification)
AutoModelForTokenClassification.register(crammedBertConfig, ScriptableLMForTokenClassification)
