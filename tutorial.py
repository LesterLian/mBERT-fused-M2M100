from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, BertTokenizer, BertModel, M2M100Model

import torch
import torch.nn.functional as F
from torch import nn

from typing import Optional, Tuple

from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Attention

from torchvision.models.resnet import resnet18


class FuseLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return


# class FusedAttention(M2M100Attention):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""
#
#     def __init__(
#             self,
#             bert_embedding,
#             bert_attention: BertAttention,
#             m2m_attention: M2M100Attention,
#     ):
#         self.bert_embedding = bert_embedding
#         self.bert_attention = bert_attention
#         self.m2m_attention = m2m_attention
#         self.fuse_layer = FuseLayer()
#
#         self.embed_dim = self.m2m_attention.embed_dim
#         self.num_heads = self.m2m_attention.num_heads
#         self.dropout = self.m2m_attention.dropout
#         self.head_dim = self.embed_dim // self.num_heads
#         self.scaling = self.head_dim ** -0.5
#         self.is_decoder = self.m2m_attentionis_decoder
#
#         self.k_proj = self.m2m_attention.k_proj
#         self.v_proj = self.m2m_attention.v_proj
#         self.q_proj = self.m2m_attention.q_proj
#         self.out_proj = self.m2m_attention.out_proj
#
#     def forward(
#             self,
#             hidden_states: torch.Tensor,
#             key_value_states: Optional[torch.Tensor] = None,
#             past_key_value: Optional[Tuple[torch.Tensor]] = None,
#             attention_mask: Optional[torch.Tensor] = None,
#             layer_head_mask: Optional[torch.Tensor] = None,
#             output_attentions: bool = False,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         """Input shape: Batch x Time x Channel"""
#         m2m_output = self.m2m_attention.forward(hidden_states, key_value_states, past_key_value, attention_mask,
#                                                 layer_head_mask, output_attentions)
#         bert_output = self.bert_attention.forward()
#         return


class FusedM2M(nn.Module):
    def __init__(self, bert: BertModel, m2m: M2M100Model, path: str = None):
        super().__init__()
        self.bert = bert
        self.m2m = m2m
        self.fuse_layer_path = path

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
            bert_input=None,
    ):
        if bert_input:
            # Get BERT embedding
            bert_output = self.bert(**bert_input).last_hidden_state
            # Get BERT attention outputs
            attention_outputs = self.bert(**bert_input, embedding_input=bert_output).attention_outputs
            # Pass in BERT attention outputs to M2M layers
            for i in range(len(attention_outputs)):
                self.m2m.encoder.layers[i].bert_attention_output = attention_outputs[i]
            # Load fuse layer
            m2m.load_state_dict(torch.load(self.fuse_layer_path))
            return
            # return self.m2m(input_ids=input_ids,
            #                 attention_mask=attention_mask,
            #                 decoder_input_ids=decoder_input_ids,
            #                 decoder_attention_mask=decoder_attention_mask,
            #                 head_mask=head_mask,
            #                 decoder_head_mask=decoder_head_mask,
            #                 encoder_outputs=encoder_outputs,
            #                 past_key_values=past_key_values,
            #                 inputs_embeds=inputs_embeds,
            #                 decoder_inputs_embeds=decoder_inputs_embeds,
            #                 labels=labels,
            #                 use_cache=use_cache,
            #                 output_attentions=output_attentions,
            #                 output_hidden_states=output_hidden_states,
            #                 return_dict=return_dict
            #                 )


# resnet = resnet18()
hi_text = "जीवन एक चॉकलेट बॉक्स की तरह है।"
chinese_text = "生活就像一盒巧克力啦啦啦。"

m2m = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# translate Hindi to French
# tokenizer.src_lang = "hi"
# encoded_hi = tokenizer(hi_text, return_tensors="pt")
# generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("fr"))
# tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# "La vie est comme une boîte de chocolat."

# translate Chinese to English
m2m_tokenizer.src_lang = "zh"
encoded_zh = m2m_tokenizer(chinese_text, return_tensors="pt")
generated_tokens = m2m.generate(**encoded_zh, forced_bos_token_id=m2m_tokenizer.get_lang_id("en"))
m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
"Life is like a box of chocolate."
# for id in encoded_zh.data['input_ids'][0]:
#     if int(id) in m2m_tokenizer.decoder:
#         print(m2m_tokenizer.decoder[int(id)])
#     else:
#         print(m2m_tokenizer.id_to_lang_token[int(id)])
print([m2m_tokenizer.decoder[int(id)] if int(id) in m2m_tokenizer.decoder else m2m_tokenizer.id_to_lang_token[int(id)]
       for id in encoded_zh.data['input_ids'][0]])

bert_type = 'bert-base-multilingual-cased'  # 'bert-base-multilingual-cased' or 'bert-large-multilingual-cased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_type)
bert = BertModel.from_pretrained(bert_type)
text = "Replace me by any text you'd like."
encoded_input = bert_tokenizer(chinese_text, return_tensors='pt')
output = bert(**encoded_input)
print([bert_tokenizer.ids_to_tokens[int(id)] for id in encoded_input.data['input_ids'][0]])

bert_vocab = bert_tokenizer.ids_to_tokens.values()
# common = [token for token in m2m_tokenizer.encoder if token not in bert_vocab]

fused_model = FusedM2M(bert, m2m)
fused_model()
print('here')
