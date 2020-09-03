export TOKENIZERS_PARALLELISM=false
export BC6PM_dir=../BC6PM
export pretrain_dir=../BioBERT-Models
export TMPDIR='./tmp'
export CUDA_VISIBLE_DEVICES='3'
python -u train.py \
--task BC6PM \
--do_train \
--do_eval \
--do_triage \
--loggerComment triage-Only \
--do_cross_valid \
--fineTune \
# --batch_size_triage 8 \
# --do_sanity_check \
# --ckpt_fold_num 4 \