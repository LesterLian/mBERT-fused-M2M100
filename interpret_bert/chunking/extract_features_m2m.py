# Updating code for M2M100 feature extraction

# extract M2M100 features for chunking related tasks
# (clustering and visualization)

import collections
import argparse
from tqdm import tqdm
import json
import random

random.seed(123)

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
# Added M2M100 imports for 828I_Multilingual Project
from transformers import M2M100Config, M2M100Model, M2M100ForConditionalGeneration, M2M100Tokenizer


class InputExample(object):
    def __init__(self, unique_id, text, span, label):
        self.unique_id = unique_id
        self.text = text
        self.span = span
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def read_examples(input_file, num_labeled_chunks, num_unlabeled_chunks):
    """Read a list of `InputExample`s from an input file."""
    # read all the chunks
    labeled_chunks, unlabeled_chunks = [], []
    cur_chunk_tokens, cur_chunk_type = [], None
    id2sent, idi, cur_tokens = {}, 0, []
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            if len(line) == 0:
                assert (len(cur_tokens) != 0)
                id2sent[idi] = cur_tokens
                cur_tokens = []
                idi += 1
            else:
                cur_tokens.append(line.split()[0])
    if len(cur_tokens) != 0:
        id2sent[idi] = cur_tokens
    idi, wi = 0, 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            if len(line) == 0:
                if len(cur_chunk_tokens) > 0:
                    assert (cur_chunk_type != None)
                    if cur_chunk_type != 'O':
                        labeled_chunks.append([idi, cur_chunk_tokens, cur_chunk_type])
                    else:
                        unlabeled_chunks.append([idi, cur_chunk_tokens, cur_chunk_type])
                    cur_chunk_tokens, cur_chunk_type = [], None
                idi += 1
                wi = 0
            else:
                token, _, tag = line.split()
                if tag[0] == 'B':
                    if len(cur_chunk_tokens) > 0:
                        assert (cur_chunk_type != None)
                        if cur_chunk_type != 'O':
                            labeled_chunks.append([idi, cur_chunk_tokens, cur_chunk_type])
                        else:
                            unlabeled_chunks.append([idi, cur_chunk_tokens, cur_chunk_type])
                        cur_chunk_tokens, cur_chunk_type = [], None
                    cur_chunk_tokens.append([token, wi])
                    cur_chunk_type = tag.split('-')[-1]
                elif tag[0] == 'I':
                    assert (len(cur_chunk_tokens) != 0)
                    assert (cur_chunk_type == tag.split('-')[-1])
                    cur_chunk_tokens.append([token, wi])
                elif tag[0] == 'O':
                    if len(cur_chunk_tokens) == 0:
                        # O is starting token
                        assert (cur_chunk_type == None)
                        cur_chunk_tokens.append([token, wi])
                        cur_chunk_type = tag.split('-')[-1]
                    else:
                        assert (cur_chunk_type != None)
                        if cur_chunk_type == 'O':
                            cur_chunk_tokens.append([token, wi])
                        else:
                            labeled_chunks.append([idi, cur_chunk_tokens, cur_chunk_type])
                            cur_chunk_tokens, cur_chunk_type = [], None
                            cur_chunk_tokens.append([token, wi])
                            cur_chunk_type = tag.split('-')[-1]
                wi += 1

    # shuffle the chunks
    random.shuffle(labeled_chunks)
    random.shuffle(unlabeled_chunks)
    labeled_chunks = labeled_chunks[0:num_labeled_chunks]
    unlabeled_chunks = unlabeled_chunks[0:num_unlabeled_chunks]

    # prepare the examples
    examples = []
    unique_id = 0
    for chunk in labeled_chunks + unlabeled_chunks:
        examples.append(InputExample(unique_id=unique_id, text=id2sent[chunk[0]], span=chunk[1], label=chunk[2]))
        unique_id += 1
    return examples


def get_max_seq_length(examples, tokenizer):
    max_seq_len = -1
    for example in examples:
        cand_tokens = tokenizer.tokenize(' '.join(example.text))
        cur_len = len(cand_tokens)
        if cur_len > max_seq_len:
            max_seq_len = cur_len
    return max_seq_len


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        cand_tokens = tokenizer.tokenize(' '.join(example.text))
        c_tokens = tokenizer(' '.join(example.text))
        # Account for [CLS] and [SEP] with "- 2" (__en__ and </s> for 828I Multilingual Project)
        if len(cand_tokens) > seq_length - 2:
            cand_tokens = cand_tokens[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        # Change BOS and SEP token for 828I Multilingual Project
        # tokens.append("[CLS]")  # Doubtful usage of CLS of BERT in M2M-100
        tokens.append('__en__')
        # tokens.append("<s>") #Doubtful choice of BOS token in M2M-100

        input_type_ids.append(0)
        for token in cand_tokens:
            tokens.append(token)
            input_type_ids.append(0)

        # tokens.append("[SEP]")  # Doubtful usage of SEP of BERT in M2M-100

        tokens.append("</s>")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def get_chunk_spans(examples, features):
    chunk_spans = []
    for example, feature in zip(examples, features):
        m2m_tokens = feature.tokens[1:]
        actual_tokens = example.text
        span_tokens = example.span
        ati, bti, a2b = 0, 0, []

        while ati < len(actual_tokens):
            start, end = bti, bti
            atok = actual_tokens[ati]
            toki = 0
            while toki < len(atok):
                btok = m2m_tokens[bti]
                if btok.startswith('##'):
                    btok = btok[2:]
                # Handle _ in M2M tokens for 828I Multilingual Project
                while btok.startswith('▁'):
                    btok = btok[1:]
                assert (btok.lower() == atok[toki:toki + len(btok)].lower())
                end = bti
                toki += len(btok)
                bti += 1
            a2b.append([start, end])
            ati += 1
        span_start, span_end = span_tokens[0][1], span_tokens[-1][1]
        chunk_spans.append([1 + a2b[span_start][0], 1 + a2b[span_end][1]])
    return chunk_spans


def save(args, model, tokenizer, device):
    # convert data to ids
    examples = read_examples(args.train_file, 3000,
                             500)  # default number of labeled and unlabeld chunks to consider are obtained from https://aclweb.org/anthology/D18-1179
    features = convert_examples_to_features(examples=examples, seq_length=2 + get_max_seq_length(examples, tokenizer),
                                            tokenizer=tokenizer)
    chunk_spans = get_chunk_spans(examples, features)

    # extract and write features
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    pbar = tqdm(total=len(examples) // args.batch_size)
    with open(args.output_file, "w", encoding='utf-8') as writer:
        for input_ids, input_mask, example_indices in eval_dataloader:
            input_ids = input_ids.to(device)
            # Modified for M2M model and tokenizer for 828I Multilingual Project
            # input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
            input_mask = input_mask.to(device)
            # all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            _, all_encoder_layers = model.encoder(input_ids, attention_mask=input_mask,
                                                  output_hidden_states=True,
                                                  return_dict=False
                                                  )
            for b, example_index in enumerate(example_indices):
                feature_info = features[example_index.item()]
                unique_id = int(feature_info.unique_id)
                example_info, chunk_info = examples[unique_id], chunk_spans[unique_id]
                output_json = collections.OrderedDict()
                output_json["linex_index"] = unique_id
                output_json["label"] = example_info.label
                output_json["tokens"] = feature_info.tokens
                output_json["chunk_start_idx"] = chunk_info[0]
                output_json["chunk_end_idx"] = chunk_info[1]
                span_start_layers, span_end_layers = [], []
                for layer_index in range(len(all_encoder_layers)):
                    layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                    span_start_layers.append([round(x.item(), 6) for x in layer_output[b][chunk_info[0]]])
                    span_end_layers.append([round(x.item(), 6) for x in layer_output[b][chunk_info[1]]])
                output_json["start_layer"] = span_start_layers
                output_json["end_layer"] = span_end_layers
                writer.write(json.dumps(output_json) + "\n")
            pbar.update(1)
    pbar.close()
    print('written features to %s' % (args.output_file))


# Updated BERT to M2M100 models in the following function for 828I Multilingual Project
def load(args):
    # The below line is not useful. Maybe deleted later
    print('loading M2M-100 model')
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    '''
    tokenizer = BertTokenizer.from_pretrained(args.m2m100_model, do_lower_case=True, cache_dir=args.cache_dir)
    model = BertModel.from_pretrained(args.m2m100_model, cache_dir=args.cache_dir)
    '''
    model = M2M100Model.from_pretrained('facebook/m2m100_418M')
    tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')
    model.to(device)
    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        required=True,
                        help="path to the train set from CoNLL-2000 chunking corpus")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="output file where the features will be written")
    parser.add_argument("--cache_dir",
                        default="/tmp",
                        type=str,
                        help="directory to cache bert pre-trained models")
    parser.add_argument("--m2m100_model",
                        default="bert-base-uncased",
                        type=str,
                        help="bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="whether not to use CUDA when available")
    parser.add_argument("--batch_size",
                        default=2,
                        type=int,
                        help="total batch size for inference")
    parser.add_argument("--num_gpus",
                        default=1,
                        type=int,
                        help="no. of gpus to use")

    args = parser.parse_args()
    print(args)
    model, tokenizer, device = load(args)
    save(args, model, tokenizer, device)


if __name__ == "__main__":
    main()
