import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net
from data_load import NerDataset, RCDataSet, TriageDataSet
import os
import json
import numpy as np
from collections import OrderedDict
import argparse
from utils import *
from torch.utils.tensorboard import SummaryWriter
import random
from tqdm import tqdm
import sys
import copy
parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action='store_true', help ='train the model')
parser.add_argument("--do_eval", action='store_true')
# parser.add_argument("--do_test", action='store_true')
parser.add_argument("--do_cross_valid", action='store_true', help = 'use training data with cross validation to train the model, if False, eval on testdata.')
parser.add_argument("--ignoreLongDocument", action='store_true', help = 'ignore the too long documents for BERT')
parser.add_argument("--train_withGNP", action='store_true', help = 'train whith the annotation found by GNormPlus')
parser.add_argument("--fineTune", action='store_true', help = 'fine tune BERT or froze it')
parser.add_argument("--saveModel", action='store_true', help = 'save the whole model when an epoch is over')
parser.add_argument("--do_sanity_check", action='store_true', help = 'use minimal sources to do sanity check' )
parser.add_argument(
    "--pretrain_dir", default='../BioBERT-Models', help = 'the directory of pretrained model')
parser.add_argument("--ckpt_fold_num", default=None, type=int, help = 'Continue the training from ckpt_fold_num if the cross valid training was aborted' )
parser.add_argument("--do_ner", action='store_true')
parser.add_argument("--do_rc", action='store_true')
parser.add_argument("--do_triage", action='store_true')
parser.add_argument("--do_fewshot", action='store_true')
parser.add_argument("--shotnum", default=10, type=int)
parser.add_argument("--do_normalization", action='store_true')
parser.add_argument("--batch_size_ner", default=16, type=int)
parser.add_argument("--batch_size_rc", default=8, type=int)
parser.add_argument("--batch_size_triage", default=8, type=int)
parser.add_argument("--lr", default=5e-5, type=float, help ='learning rate Except BERT')
parser.add_argument("--max_epoch_ner", default=10, type=int)
parser.add_argument("--max_epoch_rc", default=10, type=int)
parser.add_argument("--max_epoch_triage", default=10, type=int)
parser.add_argument("--max_epoch_joint", default=10, type=int)
parser.add_argument("--fold_num", default=10, type=int)
parser.add_argument("--seed", default=20200529, type=int)
parser.add_argument("--task", default='BC2GM')
parser.add_argument("--loggerComment")
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if args.do_sanity_check:
    args.max_epoch_ner = 3
    args.max_epoch_rc = 10
    args.max_epoch_joint = 10
    args.batch_size_ner = 3
    args.batch_size_rc = 3
    args.fold_num = 5
    args.loggerComment += '-Sanity_Check'

Fold_iter_idx = 0
logger = None
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
if not os.path.exists('outputjson'):
    os.makedirs('outputjson')

if not args.do_cross_valid:
    args.loggerComment += '-Test'
else:
    args.loggerComment += '-Train'
if args.fineTune:
    args.loggerComment += '-fineTune'
if args.ignoreLongDocument:
    args.loggerComment += '-ignoreLong'
if args.train_withGNP:
    args.loggerComment += '-GNP'
    args.weight_label = True  # because train with GNP will introduces too much negative instances')

def train_NER(model, iterator, dataset, epoch):
    model.train()
    iterator = tqdm(iterator, ncols=120)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        _y = y  # for monitoring
        model.optimizer.zero_grad()
        logits, y, _ = model.forwardNER(x, y)

        logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        model.optimizer.step()

        if i == 0 and args.do_sanity_check:
            print("=====NER sanity check======")
            print("x:", x.cpu().numpy()[0])
            print("words:", words[0])
            print("tokens:", dataset.tokenizer.convert_ids_to_tokens(
                x.cpu().numpy()[0]))
            print("y:", _y.cpu().numpy()[0])
            print("is_heads:", is_heads[0])
            print("tags:", tags[0])
            print("seqlen:", seqlens[0])
        iterator.set_description(
            f"Train NER Fold: {Fold_iter_idx}, Epoch: {epoch}, loss: {loss.item():.4f}")
        if i % 10 == 0 or args.do_sanity_check:  # monitoring
            logger.add_scalars(
                'Loss/NER', {'Fold_'+str(Fold_iter_idx): loss.item()}, (epoch-1)*len(iterator) + i)


def eval_NER(model, iterator, f, dataset, epoch):
    model.eval()
    iterator = tqdm(iterator, ncols=120)

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model.forwardNER(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
            iterator.set_description(
                f"Eval NER Fold: {Fold_iter_idx}, Epoch: {epoch}")

    # gets results and save
    with open(f, 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [dataset.idx2tag[hat] for hat in y_hat]
            assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w}\t{t}\t{p}\n")
            fout.write("\n")

    # calc metric
    y_true = np.array([line.split()[1] for line in open(
        f, 'r').read().splitlines() if len(line) > 0])
    y_pred = np.array([line.split()[2] for line in open(
        f, 'r').read().splitlines() if len(line) > 0])
    words = np.array([line.split()[0] for line in open(
        f, 'r').read().splitlines() if len(line) > 0])

    anns_true = []
    anns_pred = []

    def tags2anns(tags):
        anns = []
        flag_start = False
        for i, tag in enumerate(tags):
            if tag == 'B':
                flag_start = True
                tmp_list = [i, i]
                anns.append(tmp_list)
                continue
            if flag_start:
                if tag == 'O':
                    flag_start = False
                elif tag == 'I':
                    tmp_list[1] = i
        annSet = set()
        for ann in anns:
            annSet.add(tuple(ann))
        return annSet

    anns_true = tags2anns(y_true)
    anns_pred = tags2anns(y_pred)
    true_num = len(anns_true)
    pred_num = len(anns_pred)
    tp_num = len(anns_true & anns_pred)
    fp_num = pred_num - tp_num

    try:
        precision = tp_num / pred_num
    except ZeroDivisionError:
        precision = 0

    try:
        recall = tp_num / true_num
    except ZeroDivisionError:
        recall = 0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0

    countDict = {
        'tp': tp_num,
        'fp': fp_num,
        'true_gene_num': true_num,
        'pred_gene_num': pred_num
    }

    return precision, recall, f1, countDict


def train_RC(model, iterator, dataset, epoch):
    model.train()
    iterator = tqdm(iterator, ncols=120)
    if args.weight_label:
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.9]).cuda())
    else:
        criterion = nn.CrossEntropyLoss()

    for i, batch in enumerate(iterator):
        x, labels, guid = batch
        model.optimizer.zero_grad()
        logits, y, y_hat = model.forwardRC(x, labels)

        loss = criterion(logits, y)
        loss.backward()

        model.optimizer.step()
        if i == 0 and args.do_sanity_check:
            print("=====RC sanity check======")
            print("x:", x)
            print("x:", x['input_ids'][0])
            print("tokens:", dataset.tokenizer.convert_ids_to_tokens(
                x['input_ids'][0]))
            print("guid:", guid[0])
            print("label:", labels[0])
            print("logits:", logits[0])
            print("y:", y[0])
            print("y_hat:", y_hat[0])
        iterator.set_description(
            f"Train RC Fold: {Fold_iter_idx}, Epoch: {epoch}, loss: {loss.item():.4f}")
        if i % 10 == 0 or args.do_sanity_check:  # monitoring
            logger.add_scalars(
                'Loss/RC', {'Fold_'+str(Fold_iter_idx): loss.item()}, (epoch-1)*len(iterator) + i)


def eval_RC(model, iterator, epoch, documents_eval, documents_eval_ground_truth):
    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    iterator = tqdm(iterator, ncols=120)

    softmax = nn.Softmax(dim=1)

    docu2relations = {}
    docu2maxConf = {}
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, labels, guid = batch
            logits, y, y_hat = model.forwardRC(x, labels)

            probs = softmax(logits)
            for id_, prob, y_hat_ in zip(guid, probs, y_hat):
                pmid, gene_a, gene_b = id_.split('_')
                confidence = prob[1].item()
                if docu2relations.get(pmid) is None:
                    docu2relations[pmid] = {}
                # At least one PPIm exists
                if docu2maxConf.get(pmid, [-1])[0] < confidence:
                    docu2maxConf[pmid] = [confidence,
                                            tuple(sorted([gene_a, gene_b]))]
                # if y_hat_.item() == 1:
                docu2relations[pmid][tuple(
                    sorted([gene_a, gene_b]))] = confidence

            true_idx = y == y_hat
            false_idx = y != y_hat
            tp += (y[true_idx] == 1).sum()
            tn += (y[true_idx] == 0).sum()
            fp += (y[false_idx] == 0).sum()
            fn += (y[false_idx] == 1).sum()
            iterator.set_description(
                f"Eval RC Fold: {Fold_iter_idx}, Epoch: {epoch}")

    documents_eval_ = copy.deepcopy(documents_eval)
    for document in documents_eval_:
        document['relations'] = []
        pmid = document['id']
        if docu2relations.get(pmid) is None:
            if docu2maxConf.get(pmid) is not None:
                conf, gene_pair = docu2maxConf.get(pmid)
                docu2relations[pmid][gene_pair] = conf

        for gene_pair, conf in docu2relations.get(pmid, {}).items():
            document['relations'].append(
                {
                    "nodes": [],
                    "infons": {
                        "Gene1": gene_pair[0],
                        "Gene2": gene_pair[1],
                        "relation": "PPIm",
                        "confidence": conf
                    },
                    "id": "R0"
                }
            )

    gold_standard_ids, gold_standard_relations = JSON_Collection_Relation({'documents': documents_eval_ground_truth})
    micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1 = Classification_Performance_Relation({'documents': documents_eval_}, gold_standard_ids, gold_standard_relations)
    with open(f"outputjson/{args.loggerComment}_{epoch}.json", 'w') as f:
        json.dump({'documents': documents_eval_}, f, indent= 4 )
    try:
        precision = (tp.item())/(tp.item()+fp.item())
    except ZeroDivisionError:
        precision = 0

    try:
        recall = (tp.item())/(tp.item()+fn.item())
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2. * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0

    countDict = {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
    return precision, recall, f1, countDict, (micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1)


def train_triage(model, iterator, dataset, epoch):
    model.train()
    iterator = tqdm(iterator, ncols=120)
    criterion = nn.CrossEntropyLoss()

    for i, batch in enumerate(iterator):
        x, labels, guid = batch
        model.optimizer.zero_grad()
        logits, y, y_hat = model.forwardTriage(x, labels)

        loss = criterion(logits, y)
        loss.backward()

        model.optimizer.step()
        if i == 0 and args.do_sanity_check:
            print("=====Triage sanity check======")
            print("x:", x)
            print("x:", x['input_ids'][0])
            print("tokens:", dataset.tokenizer.convert_ids_to_tokens(
                x['input_ids'][0]))
            print("guid:", guid[0])
            print("label:", labels[0])
            print("logits:", logits[0])
            print("y:", y[0])
            print("y_hat:", y_hat[0])
        iterator.set_description(
            f"Train Triage Fold: {Fold_iter_idx}, Epoch: {epoch}, loss: {loss.item():.4f}")
        if i % 10 == 0 or args.do_sanity_check:  # monitoring
            logger.add_scalars(
                'Loss/Triage', {'Fold_'+str(Fold_iter_idx): loss.item()}, (epoch-1)*len(iterator) + i)


def eval_triage(model, iterator, epoch):
    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    iterator = tqdm(iterator, ncols=120)

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, labels, guid = batch
            logits, y, y_hat = model.forwardTriage(x, labels)
            true_idx = y == y_hat
            false_idx = y != y_hat
            tp += (y[true_idx] == 1).sum()
            tn += (y[true_idx] == 0).sum()
            fp += (y[false_idx] == 0).sum()
            fn += (y[false_idx] == 1).sum()
            iterator.set_description(
                f"Eval Triage Fold: {Fold_iter_idx}, Epoch: {epoch}")

    try:
        precision = (tp.item())/(tp.item()+fp.item())
    except ZeroDivisionError:
        precision = 0

    try:
        recall = (tp.item())/(tp.item()+fn.item())
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2. * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0

    countDict = {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

    return precision, recall, f1, countDict


def train_Joint(model, iterator_a, iterator_b, task_a, task_b, epoch):
    def train_NER_batch(model, batch):
        model.train()
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        words, x, is_heads, tags, y, seqlens = batch
        model.optimizer.zero_grad()
        logits, y, _ = model.forwardNER(x, y)
        logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()
        model.optimizer.step()
        return loss

    def train_RC_batch(model, batch):
        model.train()
        criterion = nn.CrossEntropyLoss()

        x, labels, guid = batch
        model.optimizer.zero_grad()
        logits, y, y_hat = model.forwardRC(x, labels)

        loss = criterion(logits, y)
        loss.backward()
        model.optimizer.step()
        return loss

    def train_triage_batch(model, batch):
        model.train()
        criterion = nn.CrossEntropyLoss()

        x, labels, guid = batch
        model.optimizer.zero_grad()
        logits, y, y_hat = model.forwardTriage(x, labels)

        loss = criterion(logits, y)
        loss.backward()
        model.optimizer.step()
        return loss
    trian_func = {
        'NER': train_NER_batch,
        'RC': train_RC_batch,
        'triage': train_triage_batch
    }
    a = len(iterator_a)
    b = len(iterator_b)
    ratio = a/b
    less_i, more_i = 0, 0
    if ratio < 1:
        less, more = a, b
        less_iter, more_iter = iter(iterator_a),  iter(iterator_b)
        less_task, more_task = task_a, task_b
    else:
        more, less = a, b
        more_iter, less_iter = iter(iterator_a),  iter(iterator_b)
        more_task, less_task = task_a, task_b
        ratio = 1 / ratio
    less_train, more_train = trian_func[less_task], trian_func[more_task]
    with tqdm(total=a+b, ncols=120) as pbar:
        while True:
            more_acc = 0
            while more_acc * ratio < 1 and more_i < more:
                more_acc += 1
                more_i += 1
                loss = more_train(model, next(more_iter))
                pbar.set_description(
                    f"{Fold_iter_idx} {more_task}: {more_i}/{more} {less_task}: {less_i}/{less}")
                pbar.update(1)
                if more_i % 10 == 0 or args.do_sanity_check:  # monitoring
                    logger.add_scalars(
                        f'Loss/{more_task}', {'Fold_'+str(Fold_iter_idx): loss.item()}, (epoch-1)*more + more_i)
                if more_i * ratio > less_i:
                    break
            if less_i < less:
                less_train(model, next(less_iter))
                less_i += 1
                pbar.set_description(
                    f"{Fold_iter_idx} {more_task}: {more_i}/{more} {less_task}: {less_i}/{less}")
                pbar.update(1)
                if less_i % 10 == 0 or args.do_sanity_check:  # monitoring
                    logger.add_scalars(
                        f'Loss/{less_task}', {'Fold_'+str(Fold_iter_idx): loss.item()}, (epoch-1)*less + less_i)
            if less_i >= less and more_i >= more:
                break

        assert less_i == less, str(less) + ' ' + str(less_i)
        assert more_i == more, str(more) + ' ' + str(more_i)


def saveState(fold_iter_idx, epoch, model=None):
    state = {
        'fold_iter_idx': fold_iter_idx,
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
        'cuda_rng_state_all': torch.cuda.get_rng_state_all(),
        'np_rng_state': np.random.get_state(),
        'random_rng_state': random.getstate(),
        'model': None if model is None else model.state_dict()
    }
    ckpt_dir = os.path.join(
        'checkpoints', f'{args.loggerComment}_fold_{fold_iter_idx}_epoch_{epoch}.bin')
    print("Save state at " + ckpt_dir, file=sys.stderr)
    torch.save(state, ckpt_dir)


def loadState(ckpt_fold_num, epoch = 'done', model=None):
    ckpt_dir = os.path.join(
        'checkpoints', f'{args.loggerComment}_fold_{ckpt_fold_num}_epoch_{epoch}.bin')
    print("Load state at " + ckpt_dir, file=sys.stderr)
    with open(ckpt_dir, 'rb') as f:
        state = torch.load(f)
        fold_iter_idx = state['fold_iter_idx']
        torch.set_rng_state(state['torch_rng_state'])
        torch.cuda.set_rng_state(state['cuda_rng_state']),
        torch.cuda.set_rng_state_all(state['cuda_rng_state_all']),
        np.random.set_state(state['np_rng_state'])
        random.setstate(state['random_rng_state'])
        if state.get('model') is not None and model is not None:
            model.load_state_dict(state['model'])


if __name__ == "__main__":
    for fold_iter_idx in range(args.fold_num):
        if args.do_train:
            Fold_iter_idx = fold_iter_idx
            # Prepare Documents
            if args.task.lower() == 'bc2gm':
                train_dataset = NerDataset(
                    f"../{args.task}/train.tsv", args.task, args.pretrain_dir)
                eval_dataset = NerDataset(
                    f"../{args.task}/test.tsv", args.task, args.pretrain_dir)
            elif args.task.lower() == 'bc6pm':
                if args.do_cross_valid:
                    if fold_iter_idx == 0:
                        if args.train_withGNP:
                            train_data_dir = os.path.join(os.environ.get('BC6PM_dir'), 'GNormPlus', 'withAnn-Result', 'PMtask_Relations_TrainingSet_r.json')
                        else:
                            train_data_dir = os.path.join(os.environ.get('BC6PM_dir'), 'json', 'PMtask_Relations_TrainingSet.json')
                        with open(train_data_dir) as f:
                            train_data = json.load(f)
                        documents = train_data['documents']
                        if args.do_sanity_check:
                            documents = documents[:80]
                        random.shuffle(documents)
                        fold_len = len(documents)//args.fold_num
                    eval_start = fold_iter_idx*fold_len
                    eval_end = (fold_iter_idx+1)*fold_len
                    documents_train = documents[:eval_start] + \
                        documents[eval_end:]
                    documents_eval = documents[eval_start:eval_end]
                    documents_eval_ground_truth = copy.deepcopy(documents_eval)
                    # Triage Data
                    if fold_iter_idx == 0:
                        with open(os.path.join(os.environ.get('BC6PM_dir'), 'json', 'PMtask_Triage_TrainingSet.json')) as f:
                            train_data_triage = json.load(f)
                        documents_triage = train_data_triage['documents']
                        if args.do_sanity_check:
                            documents_triage = documents_triage[:80]
                        random.shuffle(documents_triage)
                        fold_len_triage = len(documents_triage)//args.fold_num
                    eval_start_triage = fold_iter_idx*fold_len_triage
                    eval_end_triage = (fold_iter_idx+1)*fold_len_triage
                    documents_train_triage = documents_triage[:eval_start_triage] + \
                        documents_triage[eval_end_triage:]
                    documents_eval_triage = documents_triage[eval_start_triage:eval_end_triage]
                else:
                    if args.train_withGNP:
                        train_data_dir = os.path.join(os.environ.get('BC6PM_dir'), 'GNormPlus', 'withAnn-Result', 'PMtask_Relations_TrainingSet_r.json')
                    else:
                        train_data_dir = os.path.join(os.environ.get('BC6PM_dir'), 'json', 'PMtask_Relations_TrainingSet.json')
                    with open(train_data_dir) as f:
                        train_data = json.load(f)
                    documents_train = train_data['documents']
                    # with open('NER_GN/NER-GN-Test_8.json') as f:
                    with open(os.path.join(os.environ.get('BC6PM_dir'), 'GNormPlus', 'result', 'PMtask_Relations_TestSet_r.json')) as f:
                        eval_data = json.load(f)
                    documents_eval = eval_data['documents']
                    with open(os.path.join(os.environ.get('BC6PM_dir'), 'json', 'PMtask_Relations_TestSet.json')) as f:
                        eval_data_ground_truth = json.load(f)
                    documents_eval_ground_truth = eval_data_ground_truth['documents']
                    if args.ignoreLongDocument: # Ignore too long text
                        with open('pmid2tokenlen.json') as f:
                            pmid2tokenlen = json.load(f)
                        def filterLongDocu(documents):
                            documents_ = []
                            for document in documents:
                                if pmid2tokenlen.get(document['id'], 1) < 512: # some documents may not have RC instance
                                    documents_.append(document)
                            return documents_
                        documents_train = filterLongDocu(documents_train)
                        documents_eval = filterLongDocu(documents_eval)
                        documents_eval_ground_truth = filterLongDocu(documents_eval_ground_truth)
                    with open(os.path.join(os.environ.get('BC6PM_dir'), 'json', 'PMtask_Triage_TrainingSet.json')) as f:
                        train_data_triage = json.load(f)
                    documents_train_triage = train_data_triage['documents']
                    with open(os.path.join(os.environ.get('BC6PM_dir'), 'json', 'PMtask_Triage_TestSet.json')) as f:
                        eval_data_triage = json.load(f)
                    documents_eval_triage = eval_data_triage['documents']

            # Define Model
            if args.ckpt_fold_num is not None and fold_iter_idx == args.ckpt_fold_num:
                loadState(args.ckpt_fold_num)
                continue
            elif args.ckpt_fold_num is not None and fold_iter_idx < args.ckpt_fold_num:
                # print("Skip")
                continue
            else:
                # print("Init A New Model")
                model = Net(pretrained_dir=args.pretrain_dir,
                            tag_num=4,
                            lr=args.lr,
                            device=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu',
                            fineTune = args.fineTune)

            # Process Data
            if args.do_ner:
                bio_train_dir = os.path.join(os.environ.get(
                    'BC6PM_dir'), 'BIO-tag', 'train.tsv')
                bio_eval_dir = os.path.join(os.environ.get(
                    'BC6PM_dir'), 'BIO-tag', 'dev.tsv')
                documents2BIO(documents_train, bio_train_dir,
                              args.pretrain_dir)
                documents2BIO(documents_eval, bio_eval_dir,
                              args.pretrain_dir)
                train_dataset = NerDataset(
                    bio_train_dir, args.task, args.pretrain_dir)
                eval_dataset = NerDataset(
                    bio_eval_dir, args.task, args.pretrain_dir)
                # train_dataset = NerDataset(f"../BC2GM/train_dev.tsv", args.task, args.pretrain_dir)
                train_ner_iter = data.DataLoader(dataset=train_dataset,
                                                 batch_size=args.batch_size_ner,
                                                 shuffle=True,
                                                 num_workers=4,
                                                 collate_fn=train_dataset.pad)
                eval_ner_iter = data.DataLoader(dataset=eval_dataset,
                                                batch_size=args.batch_size_ner,
                                                shuffle=False,
                                                num_workers=4,
                                                collate_fn=train_dataset.pad)
            if args.do_rc:
                train_dataset_rc = RCDataSet(
                    documents_train[:args.shotnum] if args.do_fewshot else documents_train, args.pretrain_dir)
                eval_dataset_rc = RCDataSet(documents_eval, args.pretrain_dir, testData= True)
                train_rc_iter = data.DataLoader(dataset=train_dataset_rc,
                                                batch_size=args.batch_size_rc,
                                                shuffle=True,
                                                num_workers=2,
                                                collate_fn=train_dataset_rc.collate_fn
                                                )
                eval_rc_iter = data.DataLoader(dataset=eval_dataset_rc,
                                               batch_size= 2 * args.batch_size_rc,
                                               shuffle=False,
                                               num_workers=2,
                                               collate_fn=train_dataset_rc.collate_fn
                                               )
            if args.do_triage:
                train_dataset_triage = TriageDataSet(
                    documents_train_triage, args.pretrain_dir)
                eval_dataset_triage = TriageDataSet(
                    documents_eval_triage, args.pretrain_dir)
                train_triage_iter = data.DataLoader(dataset=train_dataset_triage,
                                                    batch_size=args.batch_size_triage,
                                                    shuffle=True,
                                                    num_workers=2,
                                                    collate_fn=train_dataset_triage.collate_fn
                                                    )
                eval_triage_iter = data.DataLoader(dataset=eval_dataset_triage,
                                                   batch_size=args.batch_size_triage,
                                                   shuffle=False,
                                                   num_workers=2,
                                                   collate_fn=train_dataset_triage.collate_fn
                                                   )

            logger = SummaryWriter(
                comment='-{}-Fold_{}'.format(args.loggerComment, Fold_iter_idx))

            def _logPRF1(p, r, f1, component, step):
                logger.add_scalars(
                    f'Precision/{component}', {'Fold_'+str(Fold_iter_idx): p}, step)
                logger.add_scalars(
                    f'Recall/{component}', {'Fold_'+str(Fold_iter_idx): r}, step)
                logger.add_scalars(
                    f'F1/{component}', {'Fold_'+str(Fold_iter_idx): f1}, step)
            
            # Train Model & Eval
            if args.do_ner and args.do_rc:
                for epoch in range(1, args.max_epoch_joint + 1):
                    train_Joint(model, train_ner_iter,
                                train_rc_iter, 'NER', 'RC', epoch)
                    if args.do_eval:
                        # NER EVAL
                        fname = os.path.join(
                            'checkpoints', f"{args.loggerComment}_{fold_iter_idx}_{epoch}.tsv")
                        precision, recall, f1, countDict = eval_NER(
                            model, eval_ner_iter, fname, eval_dataset, epoch)
                        documents_pred = BIO2Documents(
                            documents_eval, fname)
                        (precision_, recall_, f1_), countDict_ = evalNER(
                            documents_pred, documents_eval, checkID=False)
                        os.remove(fname)
                        # for key_ in ['tp', 'fp', 'true_gene_num', 'pred_gene_num']:
                        #     assert countDict_[key_] == countDict[key_], (countDict_[
                        #         key_], countDict[key_])

                        _logPRF1(precision, recall, f1, 'NER', epoch)

                        if args.do_normalization:
                            geneNormalization(documents_pred)
                            (precision, recall, f1), countDict = evalNER(
                                documents_pred, documents_eval, checkID=True)
                            _logPRF1(precision, recall, f1, 'NER-Norm', epoch)
                        # RC EVAL
                        precision, recall, f1, countDict, BC6_evaluation = eval_RC(
                            model, eval_rc_iter, epoch, documents_eval, documents_eval_ground_truth)

                        _logPRF1(precision, recall, f1, 'RC', epoch)
                        _logPRF1(BC6_evaluation[0], BC6_evaluation[1], BC6_evaluation[2], 'RC-BC6', epoch)
                    if args.saveModel:
                        saveState(fold_iter_idx, epoch, model)

            elif (args.do_triage and args.do_rc) or args.do_fewshot:
                for epoch in range(1, args.max_epoch_joint + 1):
                    if args.do_fewshot:
                        if args.do_triage:
                            train_triage(model, train_triage_iter,
                                         train_dataset_triage, epoch)
                            if args.do_eval:
                                precision, recall, f1, countDict = eval_triage(
                                    model, eval_triage_iter, epoch)
                                _logPRF1(precision, recall,
                                         f1, 'Triage', epoch)

                        log_itv = max(10 // args.shotnum, 1)
                        epoch_fewshot = 50 * log_itv
                        for i in range(1, epoch_fewshot+1):
                            epoch_fs = (epoch-1)*epoch_fewshot+i
                            train_RC(model, train_rc_iter,
                                     train_dataset_rc, epoch_fs)
                            if args.do_eval and i % log_itv == 0:
                                precision, recall, f1, countDict, BC6_evaluation = eval_RC(
                                    model, eval_rc_iter, i//log_itv, documents_eval, documents_eval_ground_truth)
                                _logPRF1(precision, recall, f1, 'RC',
                                         (epoch-1)*50 + i//log_itv)
                                _logPRF1(BC6_evaluation[0], BC6_evaluation[1], BC6_evaluation[2], 'RC-BC6', (epoch-1)*50 + i//log_itv)


                    else:
                        train_Joint(model, train_triage_iter,
                                    train_rc_iter, 'triage', 'RC', epoch)
                        if args.do_eval:
                            precision, recall, f1, countDict = eval_triage(
                                model, eval_triage_iter, epoch)
                            _logPRF1(precision, recall, f1, 'Triage', epoch)

                            # RC EVAL
                            precision, recall, f1, countDict, BC6_evaluation = eval_RC(
                                model, eval_rc_iter, epoch, documents_eval, documents_eval_ground_truth)
                            _logPRF1(precision, recall, f1, 'RC', epoch)
                            _logPRF1(BC6_evaluation[0], BC6_evaluation[1], BC6_evaluation[2], 'RC-BC6', epoch)
                    if args.saveModel:
                        saveState(fold_iter_idx, epoch, model)


            elif args.do_ner:
                for epoch in range(1, args.max_epoch_ner+1):
                    train_NER(model, train_ner_iter,
                              train_dataset, epoch)
                    if args.do_eval:
                        fname = os.path.join(
                            'checkpoints', f"{args.loggerComment}_{fold_iter_idx}_{epoch}.tsv")
                        precision, recall, f1, countDict = eval_NER(
                            model, eval_ner_iter, fname, eval_dataset, epoch)
                        if args.task.lower() == 'bc6pm':
                            documents_pred = BIO2Documents(
                                documents_eval, fname)
                            (precision_, recall_, f1_), countDict_ = evalNER(
                                documents_pred, documents_eval, checkID=False)
                            # for key_ in ['tp', 'fp', 'true_gene_num', 'pred_gene_num']: # it can't pass because there are some filters in `Annotation.sortAnns`  
                                # assert countDict_[key_] == countDict[key_], (countDict_[
                                                                            #  key_], countDict[key_], key_)

                            _logPRF1(precision, recall, f1, 'NER', epoch)

                            if args.do_normalization:
                                geneNormalization(documents_pred)
                                (precision, recall, f1), countDict = evalNER(
                                    documents_pred, documents_eval, checkID=True)
                                _logPRF1(precision, recall,
                                         f1, 'NER-Norm', epoch)
                            with open(f'NER_GN/{args.loggerComment}_{epoch}.json', 'w') as f:
                                json.dump(documents_pred, f, indent = 4)
                        else:
                            print(
                                f"Precision: {100*precision:.2f}, Recall: {100*recall:.2f} f1: {100*f1:.2f}")
                        os.remove(fname)
                    if args.saveModel:
                        saveState(fold_iter_idx, epoch, model)

            elif args.do_rc:
                for epoch in range(1, args.max_epoch_rc+1):
                    train_RC(model, train_rc_iter,
                             train_dataset_rc, epoch)
                    if args.do_eval:
                        precision, recall, f1, countDict, BC6_evaluation = eval_RC(
                            model, eval_rc_iter, epoch, documents_eval, documents_eval_ground_truth)
                        _logPRF1(precision, recall, f1, 'RC', epoch)
                        _logPRF1(BC6_evaluation[0], BC6_evaluation[1], BC6_evaluation[2], 'RC-BC6', epoch)
                    if args.saveModel:
                        saveState(fold_iter_idx, epoch, model)

            elif args.do_triage:
                for epoch in range(1, args.max_epoch_triage+1):
                    train_triage(model, train_triage_iter,
                                 train_dataset_triage, epoch)
                    if args.do_eval:
                        precision, recall, f1, countDict = eval_triage(
                            model, eval_triage_iter, epoch)
                        _logPRF1(precision, recall, f1, 'Triage', epoch)
            logger.close()
        if args.do_cross_valid:
            saveState(fold_iter_idx, epoch = 'done')
        else:
            break

        if args.do_sanity_check and fold_iter_idx > 1:
            break
