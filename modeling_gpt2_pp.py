import torch
from transformers import GPT2PreTrainedModel
from torch import  nn


class GPT2ForProbingViaPrompting(GPT2PreTrainedModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, gpt2):
        super().__init__(config)

        self.gpt2 = gpt2
        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.n_embd = config.n_embd
        self.match_n_embd = self.n_embd // self.match_n_head

        for param in self.gpt2.parameters():
            param.requires_grad = False

        self.prefix_len = config.prefix_len
        self.prefix_dim = config.prefix_dim
        self.prefix_drop = config.prefix_drop
        print('PrefixTuning')
        print('prefix_len: {}, prefix_dim: {}, prefix_drop: {}'.format(self.prefix_len, self.prefix_dim, self.prefix_drop))


        self.input_tokens = torch.arange(self.prefix_len).long()
        self.wte = nn.Embedding(self.prefix_len, self.n_embd)
        self.prefix_model = nn.Sequential(
            nn.Linear(self.n_embd, self.prefix_dim),
            nn.Tanh(),
            nn.Linear(self.prefix_dim, self.match_n_layer * 2 * self.n_embd))

        self.dropout = nn.Dropout(self.prefix_drop)

        self.model_parallel = False
        self.device_map = None

    def get_prefix(self, bsz, device):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.prefix_model(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) # n_layer, 2, bsz, n_head, seqlen, n_embd
        return past_key_values

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
    ):

        bsz = input_ids.shape[0]
        device = input_ids.device

        past_key_values = self.get_prefix(bsz, device)

        output = self.gpt2(input_ids=input_ids,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict)

        return output