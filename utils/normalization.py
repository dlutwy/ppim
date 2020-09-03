from eutils import QueryService
import xmltodict
import copy
import json
import tqdm
import os
import sys
from dotenv import load_dotenv, find_dotenv
if __name__ == "__main__":
    from annotation import Annotation
else:
    from .annotation import Annotation
load_dotenv(find_dotenv())

api_key = os.getenv("API_KEY", None)
email = os.getenv("email", "biocommons-dev@googlegroups.com") # default_email provied project `eutils`
cachePath = os.path.join(os.path.dirname(os.path.abspath(__file__)),"eutils-cache.db")
eqs = QueryService(email=email, cache=cachePath, api_key=api_key)

def getGeneByMention(mention):
    eqsr = eqs.esearch({'db': 'gene', 'term': mention,
                        'sort': 'relevance', 'retmax': 100})
    xmlResult = eqsr.decode('utf-8')
    dict_ = xmltodict.parse(xmlResult)
    if dict_['eSearchResult']['Count'] == '0':
        return []
    else:
        return dict_['eSearchResult']['IdList']['Id']


def getGeneByPMID(PMID):
    eqsr = eqs.esearch({'db': 'gene', 'term': PMID+'[PMID]'})
    xmlResult = eqsr.decode('utf-8')
    dict_ = xmltodict.parse(xmlResult)
    if dict_['eSearchResult']['Count'] == '0':
        return []
    else:
        return dict_['eSearchResult']['IdList']['Id']


def buildCache(documents):
    print("Build NCBI search cache")
    # PMID cache
    for docu in tqdm.tqdm(documents):
        getGeneByPMID(docu['id'])

    # Gene Mention cache
    for docu in tqdm.tqdm(documents):
        for passage in docu['passages']:
            for ann in passage['annotations']:
                getGeneByMention(ann['text'])


def geneNormalization(documents):
    print("Gene Normalization:", file = sys.stderr)
    for document in tqdm.tqdm(documents):
        ids_by_pmid = getGeneByPMID(document['id'])
        for passage in document['passages']:
            for ann in passage['annotations']:
                ids_by_mention = getGeneByMention(ann['text'])
                flag = False
                for id_by_mention in ids_by_mention:
                    if id_by_mention in ids_by_pmid:
                        flag = True
                        Annotation.setNCBIID(ann, id_by_mention)
                        break
                if not flag:
                    Annotation.setNCBIID(ann, ids_by_mention[0] if len(ids_by_mention) > 0 else 'TBD')


def evalNormalization(documents_pred, documents_true):
    tp = 0
    tbd = 0
    fp = 1
    for docu_pred, docu_true in zip(documents_pred, documents_true):
        for psg_pred, psg_true in zip(docu_pred['passages'], docu_true['passages']):
            for ann_pred, ann_true in zip(psg_pred['annotations'], psg_true['annotations']):
                if Annotation.isSame(ann_pred, ann_true, checkID = True):
                    tp += 1
                elif Annotation.getNCBIID(ann_pred) == 'TBD':
                    tbd += 1
                else:
                    fp += 1
    try:
        recall = tp/(tp+tbd+fp)
    except ZeroDivisionError:
        recall = 0
    try:
        precision = tp/(tp+fp)
    except ZeroDivisionError:
        precision = 0
    
    
    print(f"Recall {recall:.2f}\tPrecision {precision:.2f}")
    return tp, fp, tbd


if __name__ == '__main__':
    with open('BC6PM/json/PMtask_Relations_TrainingSet.json') as f:
        data = json.load(f)

    documents_true = data['documents']
    buildCache(documents_true)
    documents_pred = copy.deepcopy(documents_true)
    geneNormalization(documents_pred)
    print("Input: PMtask_Relations_TrainingSet.json, Normalized By NCBI database API")
    print(evalNormalization(documents_pred, documents_true))
    print("Normalization Component Sanity Check done")
