export TOKENIZERS_PARALLELISM=false
export BC6PM_dir=../BC6PM
export pretrain_dir=../BioBERT-Models
export TMPDIR='./tmp'
export CUDA_VISIBLE_DEVICES='2'
python -u train.py \
--task BC6PM \
--do_train \
--do_eval \
--do_rc \
--loggerComment RC-Only \
--max_epoch_rc 10 \
--fineTune \
--train_withGNP \
# --ignoreLongDocument \