from datasets import load_dataset, load_metric, DatasetDict
from torch import optim

from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration, Seq2SeqTrainingArguments, \
    DataCollatorForSeq2Seq, Seq2SeqTrainer, BertTokenizer, BertModel, M2M100Model
import numpy as np
import torch


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
            # bert_output = self.bert(**bert_input).last_hidden_state
            # Get BERT attention outputs
            # attention_outputs = self.bert(**bert_input, embedding_input=bert_output).attention_outputs
            # Pass in BERT attention outputs to M2M layers
            # for i in range(len(attention_outputs)):
            #     self.m2m.model.encoder.layers[i].bert_attention_output = attention_outputs[i]
            # self.m2m.model.encoder.layers[-1].bert_attention_output = attention_outputs[-1]
            # Load fuse layer
            if self.fuse_layer_path:
                m2m.load_state_dict(torch.load(self.fuse_layer_path))

    def forward(self, *input, **kwargs):
        self.m2m.model.encoder.layers[-1].bert_attention_output = kwargs['bert_attention_output']
        kwargs = {k: v for k, v in kwargs.items() if k in ['input_ids', 'labels']}
        return self.m2m(**kwargs)


# Load dataset
# raw_datasets = load_dataset("bible_para", lang1="af", lang2="fi",
#                             split=['train[:-3000]', 'train[-3000:-1000]', 'train[-1000:]'])
# raw_datasets = load_dataset("bible_para", lang1="af", lang2="fi", split='train').train_test_split(1000)
raw_datasets = load_dataset('wmt20_mlqe_task1', 'si-en')

# Preprocess data
source_lang = 'si'
target_lang = 'en'
max_input_length = 1024
max_target_length = 1024
m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
m2m_tokenizer.src_lang = source_lang
m2m_tokenizer.tgt_lang = target_lang
bert_type = 'bert-base-multilingual-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_type)
max_input_length_bert = 768

bert = BertModel.from_pretrained(bert_type)
for para in bert.parameters():
    para.requires_grad = False


def preprocess_m2m(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = m2m_tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with m2m_tokenizer.as_target_tokenizer():
        labels = m2m_tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_bert(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    model_inputs = bert_tokenizer(inputs, max_length=max_input_length_bert, truncation=True)
    bert_inputs = bert_tokenizer(inputs, max_length=max_input_length_bert, truncation=True, padding=True, return_tensors="pt")
    bert_output = bert(**bert_inputs).last_hidden_state
    model_inputs["bert_attention_output"] = bert(**bert_inputs, embedding_input=bert_output).attention_outputs[-1].tolist()

    return model_inputs


# Originally Written
def preprocess(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = m2m_tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True, return_tensors='pt').tolist()

    # Setup the tokenizer for targets
    with m2m_tokenizer.as_target_tokenizer():
        labels = m2m_tokenizer(targets, max_length=max_target_length, truncation=True, padding=True, return_tensors='pt')

    model_inputs['input_ids'] = model_inputs['input_ids'].tolist()
    model_inputs["labels"] = labels["input_ids"]

    # Bert
    bert_inputs = bert_tokenizer(inputs, max_length=max_input_length_bert, truncation=True, padding=True,
                                 return_tensors="pt")
    bert_output = bert(**bert_inputs).last_hidden_state
    model_inputs["bert_attention_output"] = bert(**bert_inputs, embedding_input=bert_output).attention_outputs[
        -1].tolist()
    return model_inputs


# Based on Huggingface example
column_names = raw_datasets["train"].column_names
def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = m2m_tokenizer(inputs, max_length=max_input_length, padding=False, truncation=True)

    # Setup the tokenizer for targets
    with m2m_tokenizer.as_target_tokenizer():
        labels = m2m_tokenizer(targets, max_length=max_target_length, padding=False, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    # Bert
    # Investigate if any arguments are necessary after inputs
    bert_inputs = bert_tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    bert_inputs['input_ids'] = torch.Tensor(bert_inputs['input_ids'])
    bert_inputs['attention_mask'] = torch.Tensor(bert_inputs['attention_mask'])
    bert_output = bert(**bert_inputs).last_hidden_state
    model_inputs["bert_attention_output"] = bert(**bert_inputs, embedding_input=bert_output).attention_outputs[
        -1].tolist()
    return model_inputs


def filter_none(example):
    return example["translation"][source_lang] is not None and example["translation"][target_lang] is not None


load = False
if load:
    tokenized_datasets = torch.load('data/tokenized_datasets.pt')
else:
    # Problem is due to batching
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    torch.save(tokenized_datasets, 'data/tokenized_datasets.pt')
# tokenized_datasets = raw_datasets.filter(filter_none).map(preprocess_m2m, batched=True)
# tokenized_val = raw_datasets['validation'].filter(filter_none).map(preprocess, batched=True)
# tokenized_train = raw_datasets['train'].filter(filter_none).map(preprocess, batched=True)
# bert_tokenized_datasets = raw_datasets.filter(filter_none).map(preprocess_bert, batched=True)

# Prepare models
m2m = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
for para in m2m.parameters():
    para.requires_grad = False


fused_model = FusedM2M(bert, m2m)
fuse_parameters = []
for layer in fused_model.m2m.model.encoder.layers:
    for para in layer.fuse_layer.parameters():
        para.requires_grad = True
        fuse_parameters.append(para)

# Train
# dataloader = torch.utils.data.DataLoader(tokenized_datasets['train'], batch_size=16)  # TODO: need to pad everything
# for (step, inputs) in enumerate(dataloader):
for (step, inputs) in enumerate(tokenized_datasets['train']):
    # Feed forward
    # m2m_loss = m2m(**m2m_input, labels=labels)
    loss = fused_model(**inputs)

    # Back propagation
    loss.loss.backward()

    # Step
    # optimizer = optim.Adam(fuse_parameters)
    optimizer = optim.SGD(fuse_parameters, lr=0.01)
    optimizer.step()

    # Save checkpoint
    if not (step % 500):
        state_dict = {k: v for k, v in fused_model.m2m.model.encoder.layers.state_dict().items() if 'fuse' in k}
        torch.save(state_dict, f'checkpoints/step_{step}_loss_{loss[0]:.4f}.pt')
