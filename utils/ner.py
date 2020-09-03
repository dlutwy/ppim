if __name__ == "__main__":
    from annotation import Annotation
else:
    from .annotation import Annotation
import os
import copy
import json
from transformers import BertTokenizer


def processPassageAnnotation(pred, true, countDict, checkID):
    anns_pred = pred['annotations']
    anns_true = Annotation.sortAnns(true['annotations'], TBDFilter= False)
    for ann_pred in anns_pred:
        if Annotation.gettype(ann_pred) == 'Gene':
            countDict['pred_gene_num'] += 1
        else:
            continue
        flag = False
        for ann_true in anns_true:
            if Annotation.isSame(ann_pred, ann_true, checkID):
                countDict['tp'] += 1
                flag = True
                break
        if not flag:
            countDict['fp'] += 1
            # print(ann_pred)

    countDict['true_gene_num'] += len(anns_true)
    for ann_true in anns_true:
        flag = False
        for ann_pred in anns_pred:
            if Annotation.isSame(ann_pred, ann_true, checkID):
                flag = True
                break
        if not flag:
            countDict['fn'] += 1


def nermetric(countDict):
    tp = countDict['tp']
    fp = countDict['fp']
    fn = countDict['fn']
    tn = countDict['tn']
    pred_gene_num = countDict['pred_gene_num']
    true_gene_num = countDict['true_gene_num']
    try:
        precision = tp/pred_gene_num
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp/true_gene_num
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2*precision*recall/(precision+recall)
    except ZeroDivisionError:
        f1 = 0
    # print(
        # f"precision: {precision:.2f}, recall: {recall:.2f}, f1: {f1:.2f}")
    return precision, recall, f1


def evalNER(documents_pred, documents_true, checkID=False):
    '''
    evaluate NER performance.
    :param documents_pred: `list` of document with predicted annotations.
    :param documents_true: `list` of document with ground-truth annotations.
    :param checkID: check GENE ID of annotation if `True` 
    :return: :precision, recall, f1
    '''
    countDict = {
        'pred_gene_num': 0,
        'true_gene_num': 0,
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'tn': 0
    }
    for i, docu_true in enumerate(documents_true):
        docu_pred = documents_pred[i]
        for j, passage_true in enumerate(docu_true['passages']):
            passage_pred = docu_pred['passages'][j]
            assert passage_pred['text'] == passage_true['text']
            processPassageAnnotation(
                passage_pred, passage_true, countDict, checkID)
    # print(countDict)
    return nermetric(countDict), countDict


def _ner_document_process(document, tokenizer):
    text_li = []
    tag_li = []

    def f(text):
        # str -> List[str]
        x = tokenizer.convert_ids_to_tokens(tokenizer(text)['input_ids'])[1:-1]
        tokens = []
        for token in x:
            if token[:2] == '##':
                tokens[-1] += token[2:]
            else:
                tokens.append(token)
        return tokens
    for passage in document['passages']:
        anns = passage['annotations']
        anns = Annotation.sortAnns(anns, TBDFilter= False)
        text = passage['text']
        offset_p = passage['offset']
        index = 0
        if len(anns) == 0:
            tokens = f(text)
            text_li.extend(tokens)
            tag_li.extend(['O'] * len(tokens))
        else:
            for ann in anns:
                for i, location in enumerate(ann['locations']): # unnecessary currently because of filter in `Annotation.sortAnns`
                    if i > 0:
                        print("WARNING: PMID:{}, Ann id:{} Text:{}".format(
                            document['id'], ann['id'], ann['text']))
                    offset = location['offset']
                    length = location['length']
                    tokens = f(text[index:offset-offset_p])
                    text_li.extend(tokens)
                    tag_li.extend(['O'] * len(tokens))
                    if i == len(ann['locations']) - 1:
                        mention = text[offset-offset_p: offset-offset_p+length]
                        tokens = f(mention)
                        assert mention == ann['text'], mention + '\t' + ann['text'] +'\t'+ document['id']
                        assert len(tokens) > 0
                        tag_li.extend(['B'] + ['I']*(len(tokens) - 1))
                        text_li.extend(tokens)
                    index = max(offset - offset_p + length, index)
            tokens = f(text[index:])
            text_li.extend(tokens)
            tag_li.extend(['O']*len(tokens))
    assert len(text_li) == len(tag_li)
    return text_li, tag_li


def documents2BIO(documents, BIO_file_path, pretrained_dir):
    '''
    convert documents with anns to BIO-tag File.
    :param documents: a `list` of document
    :param BIO_file_path: the path to BIO-tag file 
    :param BIO_file_path: the path to BIO-tag file 
    :param pretrained_dir: the path to pretrained_dir(BIO-BERT)
    '''

    tokenizer = BertTokenizer.from_pretrained(
        pretrained_dir, do_lower_case=False)
    with open(BIO_file_path, 'w') as f:
        for i, docu in enumerate(documents):
            text_li, tag_li = _ner_document_process(docu, tokenizer)
            for i in range(len(text_li)):
                f.write(text_li[i]+'\t'+tag_li[i]+'\n')
                if text_li[i] == '.':
                    f.write('\n')


def _readBIOResult(path):
    with open(path) as f:
        for line in f:
            if line == '\n':
                continue
            yield line.strip().split('\t')


def BIO2Documents(documents_, BIO_file_path):
    '''
    convert BIO-tag File to documents with anns.
    :param documents: a `list` of document which is the source of BIO-tag file
    :param path: the path of BIO-tag file 
    '''
    sanity_check = __name__ == '__main__'  # FOR DEBUG
    BIOGenerator = _readBIOResult(BIO_file_path)
    documents = copy.deepcopy(documents_)
    for document in documents:
        passage_offset = 0
        idx_start, idx_end = 0, 0
        flag_start = False
        for passage in document['passages']:
            anns = []
            text = passage['text']
            while True:
                if sanity_check:
                    word, tag = next(BIOGenerator)
                    # print(idx_start, word, tag, _, sep='\t')
                else:
                    word, _, tag = next(BIOGenerator)
                    # print(idx_end, word, _, tag, sep='\t')
                if word == '[UNK]':
                    word = text[idx_end:].strip()[0]
                try:
                    idx_start = idx_end + text[idx_end:].index(word)
                    idx_end = idx_start + len(word)
                except ValueError as e:
                    print(e)
                    print(word)
                    print(text)
                    raise Exception("D")

                if tag == 'B':
                    flag_start = True
                    ann = {'text': word,
                           "infons": {
                               'type': 'Gene',
                               'NCBI GENE': 'TBD'
                           },
                           'locations': [
                               {
                                   'length': len(word),
                                   'offset': passage_offset + idx_start
                               }
                           ]}
                    anns.append(ann)
                    continue
                if flag_start:
                    if tag == 'O':
                        flag_start = False
                    elif tag == 'I':
                        ann['locations'][0]['length'] = passage_offset + \
                            idx_end - ann['locations'][0]['offset']
                        start_idx = ann['locations'][0]['offset'] - \
                            passage_offset
                        ann['text'] = text[start_idx:start_idx +
                                           ann['locations'][0]['length']]
                if idx_end == len(text) or idx_end == len(text.rstrip()):
                    # passage over
                    passage_offset += (len(text) + 1)
                    idx_start, idx_end = 0, 0
                    break
            if sanity_check:
                anns_ori = Annotation.sortAnns(passage['annotations'], TBDFilter= False)
                assert len(anns) == len(anns_ori), str(document) +'\n ' + str(len(anns))+' '+ str(len(anns_ori)) +'\n' + str(anns) + '\n' + str(anns_ori)
                for i in range(len(anns)):
                    ann_my = anns[i]
                    ann_ori = anns_ori[i]
                    assert ann_my['text'] == ann_ori[
                        'text'], f"PMID:{document['id']} My ann: {ann_my['text']} \tOri ann: {ann_ori['text']}"
                    assert ann_my['locations'] == ann_ori['locations']
            passage['annotations'] = anns
            # break

    return documents

if __name__ == "__main__":
    with open('BC6PM/json/PMtask_Relations_TrainingSet.json') as f:
        data = json.load(f)
    documents = data['documents']
    documents2BIO(documents, 'BIO.tmp.tsv', 'BioBERT-Models')
    documents_pred = BIO2Documents(documents, 'BIO.tmp.tsv')
    print(evalNER(documents_pred, documents, checkID=False))
    os.remove("BIO.tmp.tsv")
    
    with open('BC6PM/GNormPlus/withAnn-Result/PMtask_Relations_TrainingSet_r.json') as f:
        data = json.load(f)
    documents = data['documents']
    documents2BIO(documents, 'BIO.tmp.tsv', 'BioBERT-Models')
    documents_pred = BIO2Documents(documents, 'BIO.tmp.tsv')
    print(evalNER(documents_pred, documents, checkID=False))
    print("NER Component Sanity Check Done")
    os.remove("BIO.tmp.tsv")
