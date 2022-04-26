from transformers import GPT2PreTrainedModel
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from allennlp.modules import scalar_mix
from torch import nn
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.file_utils import ModelOutput
from typing import Optional
from utils import STEFunction

class DiagnosticProbingOutputs(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

class GPT2ForDiagnosticProbing(GPT2PreTrainedModel):
    def __init__(self, config, gpt2):
        super().__init__(config)
        self.transformer = gpt2
        for param in self.transformer.parameters():
            param.requires_grad = False
    
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.unary = config.unary
        self.num_labels = config.num_labels
        self.mlp_dropout = config.mlp_dropout
        self.mlp_dim = config.mlp_dim
        self.use_mlp = config.use_mlp

        self.scalar_mix = scalar_mix.ScalarMix(config.n_layer, do_layer_norm=False)

        self.proj1 = nn.Conv1d(
            config.n_embd,
            config.mlp_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
        )
        self.span_extractor1 = SelfAttentiveSpanExtractor(config.mlp_dim)
        self.d_inp = self.span_extractor1.get_output_dim()
        if not self.unary:
            self.proj2 = nn.Conv1d(
                config.n_embd,
                config.mlp_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
            )
            self.span_extractor2 = SelfAttentiveSpanExtractor(config.mlp_dim)
            self.d_inp += self.span_extractor2.get_output_dim()
    
        if not self.use_mlp:
            self.classifier = nn.Sequential(
                nn.Dropout(self.mlp_dropout),
                nn.Linear(self.d_inp, self.num_labels)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.d_inp, self.mlp_dim),
                nn.Tanh(),
                nn.LayerNorm(self.mlp_dim),
                nn.Dropout(self.mlp_dropout),
                nn.Linear(self.mlp_dim, self.num_labels),
            )

        self.w = nn.Parameter(torch.empty([config.num_hidden_layers, config.num_attention_heads]))
        nn.init.xavier_uniform(self.w)
        self.num_of_heads = None
        self.use_dsp = False

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        span1s=None,
        span2s=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.use_dsp:
            head_mask = STEFunction.apply(self.w.view(-1), self.num_of_heads).view_as(self.w)
            self.apply_masks(head_mask)

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        if not self.use_mlp:
            contextual_embeddings = transformer_outputs[0]
        else:
            all_hidden_states = transformer_outputs.hidden_states[1:]
            contextual_embeddings = self.scalar_mix(all_hidden_states)
    
        span_mask = span1s[:, :, 0] != -1

        se_proj1 = self.proj1(contextual_embeddings.transpose(1, 2)).transpose(2, 1).contiguous()
        span1_emb = self.span_extractor1(se_proj1, span1s, span_indices_mask=span_mask.long())
        if not self.unary:
            se_proj2 = self.proj2(contextual_embeddings.transpose(1, 2)).transpose(2, 1).contiguous()
            span2_emb = self.span_extractor2(se_proj2, span2s, span_indices_mask=span_mask.long())
            span_emb = torch.cat([span1_emb, span2_emb], dim=2)
        else:
            span_emb = span1_emb

        logits = self.classifier(span_emb)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits[span_mask], labels[span_mask])
        
        corrections = logits[span_mask].argmax(-1) == labels[span_mask]
        correct_counts = corrections.sum()
        total_counts = len(corrections)
        accuracy = torch.tensor([[correct_counts, total_counts]], device=corrections.device)

        if not return_dict:
            output = (accuracy,)
            return ((loss,) + output) if loss is not None else output

        return DiagnosticProbingOutputs(
            loss=loss,
            logits=accuracy,
        )

    def apply_masks(self, head_mask):
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        self.transformer.apply_masks(head_mask)
    
    def get_masks(self):
        return torch.stack(self.transformer.get_masks())
    
    def apply_dsp(self, num_of_heads):
        self.num_of_heads = num_of_heads
        self.use_dsp = True