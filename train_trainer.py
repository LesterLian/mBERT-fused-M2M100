from datasets import load_dataset, load_metric, DatasetDict, load_from_disk
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration, Seq2SeqTrainingArguments, \
    DataCollatorForSeq2Seq, Seq2SeqTrainer, BertTokenizer, BertModel, M2M100Model, M2M100Config, BertConfig, \
    PretrainedConfig
import argparse
import torch


class FusedM2M(M2M100ForConditionalGeneration):
    def __init__(self, config: PretrainedConfig, bert: BertModel = None, m2m: M2M100Model = None, path: str = None,
                 bert_input=None):
        super().__init__(config)
        self.bert = bert
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
                        labels,
                        use_cache,
                        output_attentions,
                        output_hidden_states,
                        return_dict)


# Parse arguments
parser = argparse.ArgumentParser(description="mBert fused M2M100 training using traniner")
parser.add_argument(
    "--dataset_name",
    type=str,
    default='wmt20_mlqe_task1',
    help="The name of the dataset to use (from huggingface).",
)

parser.add_argument(
    "--dataset_arg",
    type=str,
    default='si-en',
    help="The argument for the dataset to use (from huggingface).",
)

parser.add_argument(
    "--predict_with_generate",
    type=bool,
    default=True,
    help="",
)

parser.add_argument(
    "--max_source_length",
    type=int,
    default=1024,
    help="The maximum total input sequence length after "
         "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
    "--max_target_length",
    type=int,
    default=1024,
    help="The maximum total sequence length for target text after "
         "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
         "during ``evaluate`` and ``predict``.",
)
parser.add_argument("--source_lang", type=str, default='si', help="Source language id for translation.")
parser.add_argument("--target_lang", type=str, default='en', help="Target language id for translation.")
parser.add_argument(
    "--max_bert_input_length",
    type=int,
    default=128,
    help=(
        "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
        " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
    ),
)
parser.add_argument(
    "--bert_type",
    type=str,
    default='bert-base-multilingual-uncased',
    help="Name or path to pretrained bert model from huggingface.",
    required=True,
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="Batch size (per device) for the training dataloader.",
)
parser.add_argument(
    "--fuse_method",
    type=int,
    default=1,
    help="The fuse method. 1: .",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=5e-5,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
parser.add_argument(
    "--max_train_steps",
    type=int,
    default=None,
    help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--lr_scheduler_type",
    type=SchedulerType,
    default="linear",
    help="The scheduler type to use.",
    choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
)
parser.add_argument(
    "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
)
parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
parser.add_argument(
    "--model_type",
    type=str,
    default=None,
    help="Model type to use if training from scratch.",
    choices=MODEL_TYPES,
)

args = parser.parse_args()

# Load dataset
raw_datasets = load_dataset(args.dataset_name, args.dataset_arg)

# Preprocess data
source_lang = args.source_lang
target_lang = args.target_lang
m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
m2m_tokenizer.src_lang = source_lang
m2m_tokenizer.tgt_lang = target_lang
bert_type = args.bert_type
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
m2m = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", config=config)
fused_model = FusedM2M(config, bert, m2m)

shared_weight = m2m.model.shared.weight.data.clone().detach()
layer_1_weight = m2m.model.encoder.layers[0].fc1.weight.data.clone().detach()
fuse_12_weight = m2m.model.encoder.layers[-1].fuse_layer.weight.data.clone().detach()


if checkpoint:
    state_dict = torch.load(f'{checkpoint}/pytorch_model.bin')
    fused_model.load_state_dict(state_dict, strict=False)
    print(f'shared: {(shared_weight == fused_model.m2m.model.shared.weight.data.detach()).all()}')
    print(f'layer 1: {(layer_1_weight == fused_model.m2m.model.encoder.layers[0].fc1.weight.data.detach()).all()}')
    print(f'fuse 12: {(fuse_12_weight == fused_model.m2m.model.encoder.layers[-1].fuse_layer.weight.data.detach()).all()}')
    # fused_model = FusedM2M.from_pretrained(state_dict=state_dict, config=config)
# else:
#     m2m = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", config=config)
#     fused_model = FusedM2M(config, bert, m2m)

modules = [fused_model.m2m.model.shared, *fused_model.m2m.model.encoder.layers[:11]]
for module in modules:
    for param in module.parameters():
        param.requires_grad = False

# Trained linear layer with original m2m
state_dict = {k: v for k, v in fused_model.m2m.model.encoder.layers[-1].state_dict().items() if 'fuse' in k}
m2m = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", config=config)
fused_model = FusedM2M(bert, m2m)
fused_model.load_state_dict(state_dict)

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
