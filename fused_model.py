import torch

from transformers import M2M100ForConditionalGeneration, PretrainedConfig, BertModel


class FusedM2M(M2M100ForConditionalGeneration):
    def __init__(self, config: PretrainedConfig, bert: BertModel = None, m2m: M2M100ForConditionalGeneration = None, path: str = None,
                 bert_input=None):
        super().__init__(config)
        self.bert = bert  # TODO: don't need
        self.m2m = m2m
        if m2m is not None:
            self.model = m2m.model
            self.base_model = m2m.base_model
        self.fuse_layer_path = path
        self.bert_input = bert_input

        if self.bert_input:
            if self.fuse_layer_path:
                m2m.load_state_dict(torch.load(self.fuse_layer_path))

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            bert_attention_output=None,
    ):
        self.m2m.model.encoder.layers[-1].bert_attention_output = bert_attention_output
        return self.m2m(input_ids,
                        attention_mask,
                        decoder_input_ids,
                        decoder_attention_mask,
                        head_mask,
                        decoder_head_mask,
                        encoder_outputs,
                        past_key_values,
                        inputs_embeds,
                        decoder_inputs_embeds,
                        labels,  # only for ConditionalGeneration, doesn't exist for M2M100Model
                        use_cache,
                        output_attentions,
                        output_hidden_states,
                        return_dict)
