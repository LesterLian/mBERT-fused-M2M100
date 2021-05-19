from datasets import load_dataset, load_metric, DatasetDict, load_from_disk
from tqdm import tqdm

from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration, Seq2SeqTrainingArguments, \
    DataCollatorForSeq2Seq, Seq2SeqTrainer, BertTokenizer, BertModel, M2M100Model, M2M100Config, BertConfig, \
    PretrainedConfig, SchedulerType
import argparse
import torch
from fused_model import FusedM2M

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
    "--load_local_dataset",
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
)
parser.add_argument(
    "--fuse_method",
    type=int,
    default=1,
    help="The fuse method. 1: 1792 linear layer for concatenated output; 2: 768 linear layer for bert output only.",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="fused-checkpoints",
    help="Directory name to save the checkpoint of fused model.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    help="Path to the checkpoint of fused model.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="Batch size (per device) for the training dataloader.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=4e-5,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
# parser.add_argument(
#     "--lr_scheduler_type",
#     type=SchedulerType,
#     default="linear",
#     help="The scheduler type to use.",
#     choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
# )
# parser.add_argument(
#     "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
# )
# parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_eval", action='store_true')
parser.add_argument("--do_generate", action='store_true')

args = parser.parse_args()

# Load dataset
# We are using 'wmt20_mlqe_task1' 'si-en'
raw_datasets = load_dataset(args.dataset_name, args.dataset_arg)

# Preprocess data
max_source_length = args.max_source_length
max_target_length = args.max_target_length
source_lang = args.source_lang
target_lang = args.target_lang
m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
m2m_tokenizer.src_lang = source_lang
m2m_tokenizer.tgt_lang = target_lang
bert_type = args.bert_type
bert_tokenizer = BertTokenizer.from_pretrained(bert_type)
max_input_length_bert = 51  # Get from tokenize inputs with bert
fuse_method = args.fuse_method
checkpoint = args.checkpoint

# Load BERT for preprocessing the BERT attention output
bert = BertModel.from_pretrained(bert_type)
for para in bert.parameters():
    para.requires_grad = False


def preprocess(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = m2m_tokenizer(inputs, max_length=args.max_bert_input_length, truncation=True)

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


# Preprocess data or load from local file
if args.load_local_dataset:
    tokenized_datasets = load_from_disk("data")
else:
    tokenized_datasets = raw_datasets.filter(filter_none).map(preprocess, batched=True)
    tokenized_datasets.save_to_disk("data")

# Prepare models
config = M2M100Config.from_pretrained("facebook/m2m100_418M")
config.method = fuse_method
m2m = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", config=config)
fused_model = FusedM2M(config, bert, m2m)

# DEBUG: Check the weight of layers
# shared_weight = m2m.model.shared.weight.data.clone().detach()
# layer_1_weight = m2m.model.encoder.layers[0].fc1.weight.data.clone().detach()
# fuse_12_weight = m2m.model.encoder.layers[-1].fuse_layer.weight.data.clone().detach()


# Load state dict from local checkpoint
if checkpoint:
    state_dict = torch.load(f'{checkpoint}/pytorch_model.bin')
    state_dict = {k: v for k, v in state_dict.items() if 'fuse' in k}  # load linear layer only
    fused_model.load_state_dict(state_dict, strict=False)
    # DEBUG: Check the weight of layers
    # print(f'shared: {(shared_weight == fused_model.model.shared.weight.data.detach()).all()}')
    # print(f'layer 1: {(layer_1_weight == fused_model.model.encoder.layers[0].fc1.weight.data.detach()).all()}')
    # print(f'fuse 12: {(fuse_12_weight == fused_model.model.encoder.layers[-1].fuse_layer.weight.data.detach()).all()}')

# Freeze M2M layers before 12th encoder layer
modules = [fused_model.model.shared, *fused_model.model.encoder.layers[:11]]
for module in modules:
    for param in module.parameters():
        param.requires_grad = False

# Train
batch_size = args.batch_size
trainer_args = Seq2SeqTrainingArguments(
    args.checkpoint_path,
    evaluation_strategy="steps",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=args.weight_decay,
    save_total_limit=3,
    num_train_epochs=args.num_train_epochs,
    predict_with_generate=True,
    fp16=True,

)
data_collator = DataCollatorForSeq2Seq(m2m_tokenizer, model=fused_model)

trainer = Seq2SeqTrainer(
    fused_model,
    trainer_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=m2m_tokenizer,
)
if args.do_train:
    trainer.train()
if args.do_eval:
    fused_model.eval()
    print(trainer.evaluate())

# Generation
if args.do_generate:
    fused_model.eval()
    orig_en_data = []
    model_prediction = []

    for example in tqdm(raw_datasets['test']):
        si_text = example["translation"][source_lang]
        en_text = example["translation"][target_lang]
        model_inputs = m2m_tokenizer(si_text, return_tensors='pt')
        model_inputs = model_inputs.to('cuda')
        bert_inputs = bert_tokenizer(si_text, max_length=max_input_length_bert, truncation=True, padding='max_length',
                                     return_tensors="pt")
        bert_inputs = bert_inputs.to('cuda')
        bert_output = bert(**bert_inputs).last_hidden_state
        model_inputs["bert_attention_output"] = bert(**bert_inputs, embedding_input=bert_output).attention_outputs[-1]
        generated_tokens = fused_model.generate(**model_inputs, forced_bos_token_id=m2m_tokenizer.get_lang_id("en"))
        decoded_preds = m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = en_text
        # print(en_text)
        # print(decoded_preds[0])
        orig_en_data.append(en_text)
        model_prediction.append(decoded_preds[0])

    with open('reference_new.en', 'w', encoding='utf-8') as label_file:
        label_file.writelines(orig_en_data)
    with open('prediction_new.en', 'w', encoding='utf-8') as pred_file:
        pred_file.writelines(model_prediction)
