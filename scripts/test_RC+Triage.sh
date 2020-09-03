export TOKENIZERS_PARALLELISM=false
export BC6PM_dir=../BC6PM
export pretrain_dir=../BioBERT-Models
export TMPDIR='./tmp'
export CUDA_VISIBLE_DEVICES='3'
python -u train.py \
--task BC6PM \
--do_train \
--do_eval \
--do_rc \
--loggerComment Joint-RC-triage \
--max_epoch_joint 10 \
--fineTune \
--train_withGNP \
--weight_label \
--do_triage \