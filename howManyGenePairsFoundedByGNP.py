import torch
import os
import sys
import numpy as np
import random
import time
import json
from utils import geneNormalization
from utils import Annotation
def getGroundTruthRelation(documents):
    relationsGroundTruth = set()
    for document in documents:
        for relation in document['relations']:
            gene_a, gene_b = sorted([relation['infons']['Gene1'], relation['infons']['Gene2']])
            guid = "{}_{}_{}".format(document['id'], gene_a, gene_b)
            relationsGroundTruth.add(guid)
    return relationsGroundTruth

def evalNERAndGN(documents, groundTruthRelation): 
    relations = set()
    for document in documents:
        anns = []
        for passage in document['passages']:
            for ann in passage['annotations']:
                if ann['infons']['type'] == 'Gene':
                    id_ = Annotation.getNCBIID(ann)
                    anns.append(id_)
        for gene_a in anns:
            for gene_b in anns:
                ga, gb = sorted([gene_a, gene_b])
                guid = "{}_{}_{}".format(document['id'], ga, gb)
                relations.add(guid)

    tp = 0
    for guid in relations:
        if guid in groundTruthRelation:
            tp += 1

    recall = tp / len(groundTruthRelation)
    f1 = 2* 1*recall/(recall+1)
    print("F1: {:.2f}\tRecall: {:.2f}".format(100*f1, 100*recall))

fileName = 'PMtask_Relations_TestSet.json'
# fileName = 'PMtask_Relations_TrainingSet.json'
with open(os.path.join('../BC6PM', 'json', fileName)) as f:
    test_data = json.load(f)
with open(os.path.join('../BC6PM', 'GNormPlus', 'result', fileName)) as f:
    gnorm = json.load(f)
    pred_data = gnorm['documents']

print("If the component of RC can get 100% precision")
groundTruth = getGroundTruthRelation(test_data['documents'])
evalNERAndGN(pred_data, groundTruth)
geneNormalization(pred_data)
evalNERAndGN(pred_data, groundTruth)

