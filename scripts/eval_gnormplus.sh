export BC6PM_dir=../BC6PM
# Evaluate the NER and NEN component of GNP
python -u howManyGeneFoundedByGNP.py \
--dataset train

python -u howManyGeneFoundedByGNP.py \
--dataset test
