import json
import os
from utils import evalNER
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='train')
args = parser.parse_args()
if args.dataset == 'test':
    fname = 'PMtask_Relations_TestSet.json'
elif args.dataset == 'train':
    fname = 'PMtask_Relations_TrainingSet.json'

with open(os.path.join('../BC6PM', 'json', fname)) as f:
    data_true = json.load(f)
with open(os.path.join('../BC6PM', 'GNormPlus', 'result', fname)) as f:
    data_pred = json.load(f)

print("Eval on " + args.dataset)
print("Before Normalization")
(precision, recall, f1), countDict = evalNER(
                                data_pred['documents'], data_true['documents'])
print("precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(precision, recall, f1))
print("After Normalization")
(precision, recall, f1), countDict = evalNER(
                                data_pred['documents'], data_true['documents'], checkID=True)
print("precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(precision, recall, f1))
# print(countDict)