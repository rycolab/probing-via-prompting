# probing-via-prompting
This repository is in accompany with the paper: [Probing via Prompting]().

## Dependencies
- python 3.8.5
- pytorch 1.7.1+cu110

## Setup
Install required packages:
```
pip install -r requirements.txt
```

## Data Prcoessing
1. Process your OntoNotes data with the [script](https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO)
2. Extract all tasks:
```
python extract_ontonotes_all.py --ontonotes /path/to/conll-formatted-ontonotes-5.0 -o ontonotes
```
This will create two folders under `ontonotes/`, one for diagnostic probing (DP), one for probing via prompting (PP).

## Probing via Prompting
```
export task=
python run_pp.py \
    --num_train_epochs 1.0 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gpt2_name_or_path gpt2 \
    --data_dir ontonotes/pp/ \
    --task $task \
    --output_dir outputs/pp/$task/ \
    --overwrite_output_dir \
    --use_fast_tokenizer False \
    --cache_dir cache/\
    --save_strategy no \
    --prefix_len 200
```
`task` can be any one of `["pos", "const", "coref", "ner", "srl", "pos_control"]`.

If you want to experiment on the random model, replace `--gpt2_name_or_path gpt2` with     
```
    --config_name gpt2 \
    --tokenizer_name gpt2 \
```

To prune attention heads for analysis, use `--do_prune`:
```
export task=
python run_pp.py \
    --num_train_epochs 1.0 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gpt2_name_or_path gpt2 \
    --data_dir ontonotes/pp/ \
    --task $task \
    --output_dir outputs/pp/pruning/$task/ \
    --overwrite_output_dir \
    --use_fast_tokenizer False \
    --cache_dir cache/ \
    --save_strategy no \
    --prefix_len 200 \
    --do_prune \
    --num_of_heads 96 \
    --pruning_lr 0.1 \
    --seed 0
```

## Diagnostic Probing
Multi-layer perceptron (MLP) probe:
```
export task=
python run_dp.py \
    --num_train_epochs 1.0 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gpt2_name_or_path gpt2 \
    --data_dir ontonotes/dp/ \
    --task $task \
    --output_dir outputs/dp/mlp/$task/ \
    --overwrite_output_dir \
    --cache_dir cache/\
    --save_strategy no 
```
Please note that DP (MLP) does not support multi-gpus due to the incompatibility between `nn.ParameterList` in AllenNLP's `ScalarMix` and `DataParallel`.

You can use linear regression (LR) probe instead by setting `--use_mlp False`:
```
export task=
python run_dp.py \
    --num_train_epochs 1.0 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gpt2_name_or_path gpt2 \
    --data_dir ontonotes/dp/ \
    --task $task \
    --output_dir outputs/dp/lr/$task/ \
    --overwrite_output_dir \
    --cache_dir cache/\
    --save_strategy no \
    --mlp_dropout 0.1 \
    --use_mlp False 
```

DP (LR) also supports head pruning:
```
export task=
python run_dp.py \
    --num_train_epochs 1.0 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gpt2_name_or_path gpt2 \
    --data_dir ontonotes/dp/ \
    --task $task \
    --output_dir outputs/dp/lr/pruning/$task/ \
    --overwrite_output_dir \
    --cache_dir cache/\
    --save_strategy no \
    --mlp_dropout 0.1 \
    --use_mlp False \
    --do_prune \
    --num_of_heads 96 \
    --pruning_lr 0.1 \
    --seed 0
```

## Amnesic Probing 
To evaluate language modeling loss when the essential heads stored in `/path/to/head_mask` are pruned, run
```
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --do_eval \
    --output_dir outputs/lm/ \
    --overwrite_output_dir \
    --per_device_eval_batch_size 32 \
    --cache_dir cache/ \
    --head_mask_path /path/to/head_mask
```