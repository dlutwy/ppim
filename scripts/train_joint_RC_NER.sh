export TOKENIZERS_PARALLELISM=false
export BC6PM_dir=../BC6PM
export pretrain_dir=../BioBERT-Models
export TMPDIR='./tmp'
export CUDA_VISIBLE_DEVICES='1'
python -u train.py \
--task BC6PM \
--do_train \
--do_eval \
--do_ner \
--do_rc \
--loggerComment Joint-RC-NER \
--fineTune \
--do_cross_valid \
# --train_withGNP \