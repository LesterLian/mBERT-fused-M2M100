from datasets import load_dataset, DatasetDict, Dataset
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, BertTokenizer, BertModel, M2M100Model, \
    M2M100Config

import torch
import torch.optim as optim

# from utils import *


# class FuseLayer(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return


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


class FusedM2M(M2M100ForConditionalGeneration):
    def __init__(self, bert: BertModel, m2m: M2M100Model, path: str = None, bert_input=None):
        super().__init__(m2m.config)
        self.bert = bert
        self.m2m = m2m
        self.model = m2m.model
        self.base_model = m2m.base_model
        self.fuse_layer_path = path
        self.bert_input = bert_input

        if self.bert_input:
            # Get BERT embedding
            bert_output = self.bert(**bert_input).last_hidden_state
            # Get BERT attention outputs
            attention_outputs = self.bert(**bert_input, embedding_input=bert_output).attention_outputs
            # Pass in BERT attention outputs to M2M layers
            for i in range(len(attention_outputs)):
                self.m2m.model.encoder.layers[i].bert_attention_output = attention_outputs[i]
            # Load fuse layer
            if self.fuse_layer_path:
                m2m.load_state_dict(torch.load(self.fuse_layer_path))

    def forward(self, *input, **kwargs):
        return self.m2m(*input, **kwargs)


hi_text = "जीवन एक चॉकलेट बॉक्स की तरह है।"
zh_text = "生活就像一盒巧克力。"
# tgt_text = 'Life is like a box of chocolates.'
my_text = "ပြင်သစ် နိုင်ငံ ပါရီ မြို့ ပါ့ဒက်စ် ပရင့်စက် ၌ ၂၀၀၇ ခုနှစ် ရပ်ဘီ ကမ္ဘာ့ ဖလား တွင် အီတလီ သည် ပေါ်တူဂီ ကို ၃၁ - ၅ ဂိုး ဖြင့် ရေကူး ကန် စီ တွင် ရှုံးနိမ့် သွား ပါ သည် ။"
en_text = 'Italy have defeated Portugal 31-5 in Pool C of the 2007 Rugby World Cup at Parc des Princes, Paris, France.'
ja_text = 'フランスのパリ、パルク・デ・プランスで行われた2007年ラグビーワールドカップのプールCで、イタリアは31対5でポルトガルを下した。'

print('original sentence:')
print(hi_text)
print(zh_text)

m2m = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
for para in m2m.parameters():
    para.requires_grad = False

# translate Chinese to English
m2m_tokenizer.src_lang = "en"
m2m_input = m2m_tokenizer(en_text, return_tensors="pt")

modules = [m2m.model.shared, *m2m.model.encoder.layers[:11]] #Replace 5 by what you want
# for module in modules:
#     for param in module.parameters():
#         param.requires_grad = False
print(m2m.model.shared.weight.requires_grad)
print(m2m.model.encoder.layers[0].fc2.weight.requires_grad)
print(m2m.model.encoder.layers[3].fc2.weight.requires_grad)
print(m2m.model.encoder.layers[6].fc2.weight.requires_grad)
print(m2m.model.encoder.layers[11].fc2.weight.requires_grad)

generated_tokens = m2m.generate(**m2m_input, forced_bos_token_id=m2m_tokenizer.get_lang_id("ja"))
print('M2M result:')
print(m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
print([m2m_tokenizer.decoder[int(id)] if int(id) in m2m_tokenizer.decoder else m2m_tokenizer.id_to_lang_token[int(id)]
       for id in m2m_input.data['input_ids'][0]])

bert_type = 'bert-base-multilingual-cased'  # 'bert-base-multilingual-cased' or 'bert-large-multilingual-cased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_type)
bert = BertModel.from_pretrained(bert_type)
for para in bert.parameters():
    para.requires_grad = False
bert_input = bert_tokenizer(zh_text, return_tensors='pt')

# output = bert(**encoded_input)
# print([bert_tokenizer.ids_to_tokens[int(id)] for id in bert_input.data['input_ids'][0]])

# bert_vocab = bert_tokenizer.ids_to_tokens.values()
# common = [token for token in m2m_tokenizer.encoder if token not in bert_vocab]

PATH = './checkpoints'
Train = True
checkpoint = PATH + '/loss_0.8885.pt'
checkpoint = None
# Initialize Fused model
fused_model = FusedM2M(bert, m2m, bert_input=bert_input)
if checkpoint:
    print('Initialize from checkpoint')
    state_dict = torch.load(checkpoint)
    print(f'Weight of layer 0 before loading:\n{fused_model.m2m.model.encoder.layers[0].fuse_layer.weight}')
    fused_model.m2m.model.encoder.layers.load_state_dict(state_dict, strict=False)
    print(f'Weight of layer 0 after loading:\n{fused_model.m2m.model.encoder.layers[0].fuse_layer.weight}')

# Fused model generation
generated_tokens = fused_model.generate(**m2m_input, forced_bos_token_id=m2m_tokenizer.get_lang_id("en"))
print('Fused model result:')
print(m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

# Train
if Train:
    # Load dataset
    # sample_data = load_dataset("wmt16", "ro-en")
    # raw_dict = {}
    # for split in ['train', 'dev', 'test']:
    #     en_data = read_file(f'data/{split}.alt.en')
    #     my_data = read_file(f'data/{split}.alt.my')
    #     raw_dict[split] = Dataset.from_dict({'translation': [{'en': e, 'my': m} for e, m in zip(en_data, my_data)]})
    #
    # # Preprocess dataset
    # def preprocess(examples):
    #     inputs = [ex['en'] for ex in examples["translation"]]
    #     targets = [ex['my'] for ex in examples["translation"]]
    #     model_inputs = m2m_tokenizer(inputs)
    #
    #     # Setup the tokenizer for targets
    #     with m2m_tokenizer.as_target_tokenizer():
    #         labels = m2m_tokenizer(targets)  # (targets, max_length=max_target_length, truncation=True)
    #
    #     model_inputs["labels"] = labels["input_ids"]
    #     return model_inputs
    #
    # m2m_tokenizer.src_lang = "en"
    # m2m_tokenizer.tgt_lang = "my"
    # for split in raw_dict:
    #     raw_dict[split] = raw_dict[split].map(preprocess, batched=True)

    # for src_example, tgt_example in zip(my_dataset, en_dataset):
    #     preprocess_function(src_example, tgt_example)
    # fused_inputs = preprocess_function(my_dataset['train'], en_dataset['train'])

    # Set grad to True for fuse layer
    fuse_parameters = []
    for layer in fused_model.m2m.model.encoder.layers[-1:]:
        for para in layer.fuse_layer.parameters():
            para.requires_grad = True
            fuse_parameters.append(para)

    # Get the target embeddings
    m2m_tokenizer.tgt_lang = 'zh'
    with m2m_tokenizer.as_target_tokenizer():
        labels = m2m_tokenizer(hi_text, return_tensors="pt").input_ids

    # Feed forward
    # m2m_loss = m2m(**m2m_input, labels=labels)
    fused_inputs = m2m_tokenizer(hi_text, return_tensors="pt")
    loss = fused_model(**fused_inputs, labels=labels)  # TODO: losses are exact same

    # Back propagation
    loss.loss.backward()

    # Step
    # optimizer = optim.Adam(fuse_parameters)
    optimizer = optim.SGD(fuse_parameters, lr=0.01)
    optimizer.step()

    # Save checkpoint
    state_dict = {k: v for k, v in fused_model.m2m.model.encoder.layers.state_dict().items() if 'fuse' in k}
    torch.save(state_dict, f'{PATH}/loss_{loss[0]:.4f}.pt')
