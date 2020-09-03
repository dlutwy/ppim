export TOKENIZERS_PARALLELISM=false
export BC6PM_dir=../BC6PM
export pretrain_dir=../BioBERT-Models
export TMPDIR='./tmp'
export CUDA_VISIBLE_DEVICES='2'
python -u train.py \
--task BC6PM \
--do_train \
--do_eval \
--do_triage \
--do_rc \
--loggerComment Joint-RC-triage \
--fineTune \
--do_cross_valid \
# --train_withGNP \