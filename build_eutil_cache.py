import json
import os
import copy
from utils import geneNormalization, buildCache, evalNormalization

with open(os.path.join(os.environ.get('BC6PM_dir'), 'json', 'PMtask_Relations_TrainingSet.json')) as f:
    data = json.load(f)

documents_true = data['documents']
buildCache(documents_true)
documents_pred = copy.deepcopy(documents_true)
geneNormalization(documents_pred)
print("Input: PMtask_Relations_TrainingSet.json, Normalized By NCBI database API")
evalNormalization(documents_pred, documents_true)