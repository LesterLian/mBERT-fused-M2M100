from datasets import load_dataset, load_metric, DatasetDict, load_from_disk
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration, Seq2SeqTrainingArguments, \
    DataCollatorForSeq2Seq, Seq2SeqTrainer, BertTokenizer, BertModel, M2M100Model, M2M100Config, BertConfig
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
                        labels,
                        use_cache,
                        output_attentions,
                        output_hidden_states,
                        return_dict)


# Load dataset
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
max_input_length_bert = 51  # Get from tokenize inputs with bert
fuse_method = 1
checkpoint = "fused-checkpoints/checkpoint-7000"

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
    bert_inputs = bert_tokenizer(inputs, max_length=max_input_length_bert, truncation=True, padding=True,
                                 return_tensors="pt")
    bert_output = bert(**bert_inputs).last_hidden_state
    model_inputs["bert_attention_output"] = bert(**bert_inputs, embedding_input=bert_output).attention_outputs[
        -1].tolist()

    return model_inputs


def preprocess(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = m2m_tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with m2m_tokenizer.as_target_tokenizer():
        labels = m2m_tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    # Bert
    bert_inputs = bert_tokenizer(inputs, max_length=max_input_length_bert, truncation=True, padding='max_length',
                                 return_tensors="pt")
    bert_output = bert(**bert_inputs).last_hidden_state
    model_inputs["bert_attention_output"] = bert(**bert_inputs, embedding_input=bert_output).attention_outputs[
        -1].tolist()
    return model_inputs


def filter_none(example):
    return example["translation"][source_lang] is not None and example["translation"][target_lang] is not None


load = True
if load:
    # tokenized_datasets = torch.load('data/tokenized_datasets.pt')
    tokenized_datasets = load_from_disk("data")
else:
    tokenized_datasets = raw_datasets.filter(filter_none).map(preprocess, batched=True)
    tokenized_datasets.save_to_disk("data")
    # torch.save(tokenized_datasets, 'data/tokenized_datasets.pt')
# Prepare models
config = M2M100Config.from_pretrained("facebook/m2m100_418M")
config.method = fuse_method
if checkpoint:
    m2m = M2M100ForConditionalGeneration.from_pretrained(checkpoint, config=config)
else:
    m2m = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", config=config)

modules = [m2m.model.shared, *m2m.model.encoder.layers[:11]]
for module in modules:
    for param in module.parameters():
        param.requires_grad = False

fused_model = FusedM2M(bert, m2m)

batch_size = 4
args = Seq2SeqTrainingArguments(
    "fused-checkpoints-2",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True,

)
data_collator = DataCollatorForSeq2Seq(m2m_tokenizer, model=fused_model)

trainer = Seq2SeqTrainer(
    fused_model,
    args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=m2m_tokenizer,
)
trainer.train()
