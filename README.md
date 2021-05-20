# BERT based Low Resource Multilingual Neural Machine Translation

## Train
Our Fused Model is trained on Google Colab with two Intel Xeon CPU core at 2.30GHz 
and one Tesla T4/P8 GPU or equivalent sets of hardware.  
The hyper-perameters for the reported results are
- Learning Rate: 4e-5
- Weight Decay: 0.01
- LR Scheduler: linear
- Epoch: 5
- Batch Size: 8
### Setup
```bash
# Download transformers 4.4.2
wget https://github.com/huggingface/transformers/archive/refs/tags/v4.4.2.zip
unzip v4.4.2.zip
cp -rf transformers-4.4.2/src/transformers/ ./
rm -rf transformers-4.4.2 v4.4.2.zip
# Replace modeling file with our modified version
git clone https://github.com/LesterLian/mBERT-fused-M2M100.git
cp -f mBERT-fused-M2M100/* ./
cp -rf mBERT-fused-M2M100/transformers ./
# Install dependencies
pip install datasets==1.5.0 sacremoses==0.0.43 sentencepiece transformers==4.4.2
```

### Train/Evaluate/Generate
The following code will replicate our training and generation process:
```bash
python train_trainer.py --do_train --do_eval --do_generate
```
We can load a trained checkpoint of the Fused Model like this:
```bash
python train_trainer.py --do_generate --checkpoint checkpoint_path
```
To use different hyper-parameters or datasets, look at the supported arguments by running:
```bash
D:/Users/lianz/Coding/Projects/cmsc828i/train_trainer.py -h
# The output is
#usage: train_trainer.py [-h] [--dataset_name DATASET_NAME]
#                        [--dataset_arg DATASET_ARG]
#                        [--load_local_dataset LOAD_LOCAL_DATASET]
#                        [--max_source_length MAX_SOURCE_LENGTH]
#                        [--max_target_length MAX_TARGET_LENGTH]
#                        [--source_lang SOURCE_LANG]
#                        [--target_lang TARGET_LANG]
#                        [--max_bert_input_length MAX_BERT_INPUT_LENGTH]
#                        [--bert_type BERT_TYPE] [--fuse_method FUSE_METHOD]
#                        [--checkpoint_path CHECKPOINT_PATH]
#                        [--checkpoint CHECKPOINT] [--batch_size BATCH_SIZE]
#                        [--learning_rate LEARNING_RATE]
#                        [--weight_decay WEIGHT_DECAY]
#                        [--num_train_epochs NUM_TRAIN_EPOCHS] [--do_train]
#                        [--do_eval] [--do_generate]
```

#### Evaluation Metrics
```bash
#chrF metric: 
git clone https://github.com/m-popovic/chrF
#characTER metric: 
git clone https://github.com/rwth-i6/CharacTER
```
All the evaluation are scripted in scores.ipynb

## Phrasal Analysis of Fused Model
### Setup
```bash
git clone https://github.com/ganeshjawahar/interpret_bert.git
cp -rf interpret_bert/chunking/* ./
cp -rf mBERT-fused-M2M100/interpret_bert/chunking/* ./
# Download labeled data
wget https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz
gunzip train.txt.gz
```
### Extract Features
```bash
python extract_features.py --train_file train.txt --output_file chunking_rep.json --batch_size 2
python extract_features_mBERT.py --train_file train.txt --output_file chunking_rep.json --batch_size 2
python extract_features_m2m.py --train_file train.txt --output_file chunking_rep.json --batch_size 2
python extract_features_fused.py --checkpoint checkpoint_path/pytorch_model.bin --train_file train.txt --output_file chunking_rep.json --batch_size 2
```
## Data & Results

Our tokenized datasets is in data folder.

Our Phrasal Analysis Visualization results are in results/Phrasal_Syntax_Visualization folder with results for mBERT, M2M-100 and mBERT fused M2M-100 model.

Our Sinhala-English low resource NMT results are in results/Phrasal_Syntax_Visualization folder.
