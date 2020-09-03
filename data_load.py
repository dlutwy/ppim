import numpy as np 
from torch.utils import data 
import torch 
from transformers import BertTokenizer
from transformers.data.processors.utils import InputExample
import re
from utils import Annotation
class NerDataset(data.Dataset):
    def __init__(self, path, task_name, pretrained_dir):
        self.VOCAB_DICT = {
            'bc5cdr': ('<PAD>', 'B-Chemical', 'O', 'B-Disease' , 'I-Disease', 'I-Chemical'),
            'bc2gm': ('<PAD>', 'B', 'I', 'O'),
            'bc6pm': ('<PAD>', 'B', 'I', 'O'),
            'bionlp3g' : ('<PAD>', 'B-Amino_acid', 'B-Anatomical_system', 'B-Cancer', 'B-Cell', 
                        'B-Cellular_component', 'B-Developing_anatomical_structure', 'B-Gene_or_gene_product', 
                        'B-Immaterial_anatomical_entity', 'B-Multi-tissue_structure', 'B-Organ', 'B-Organism', 
                        'B-Organism_subdivision', 'B-Organism_substance', 'B-Pathological_formation', 
                        'B-Simple_chemical', 'B-Tissue', 'I-Amino_acid', 'I-Anatomical_system', 'I-Cancer', 
                        'I-Cell', 'I-Cellular_component', 'I-Developing_anatomical_structure', 'I-Gene_or_gene_product', 
                        'I-Immaterial_anatomical_entity', 'I-Multi-tissue_structure', 'I-Organ', 'I-Organism', 
                        'I-Organism_subdivision', 'I-Organism_substance', 'I-Pathological_formation', 'I-Simple_chemical', 
                        'I-Tissue', 'O')
        }
        self.VOCAB = self.VOCAB_DICT[task_name.lower()]
        self.tag2idx = {v:k for k,v in enumerate(self.VOCAB)}
        self.idx2tag = {k:v for k,v in enumerate(self.VOCAB)}

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=False)
        instances = open(path).read().strip().split('\n\n')
        sents = []
        tags_li = []
        for entry in instances:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<PAD>"] + tags + ["<PAD>"])
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)


    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

        # We give credits only to the first piece.
        x, y = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = self.tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [self.tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


    def pad(self, batch):
        '''Pads to the longest sample'''
        f = lambda x: [sample[x] for sample in batch]
        words = f(0)
        is_heads = f(2)
        tags = f(3)
        seqlens = f(-1)
        maxlen = np.array(seqlens).max()

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
        x = f(1, maxlen)
        y = f(-2, maxlen)


        f = torch.LongTensor

        return words, f(x), is_heads, tags, f(y), seqlens


class RCDataSet(data.Dataset):
    label_map = {'0': 0, '1': 1}
    def __init__(self, documents, pretrained_dir, testData = False):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=False)
        self.examples = []
        self.label_1_count = 0
        self.label_0_count = 0
        self.testData = testData
        for i, document in enumerate(documents):
            self.examples.extend(self.__create_examples(
                *self.__train_document_process(document)))
        print("Examples_total_num {}.\tLabel_0_num: {}\tLabel_1_num: {}".format(
            len(self.examples), self.label_0_count, self.label_1_count))
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __train_document_process(self, document):
        pmid = document['id']
        relations = set()
        genes = set()
        text_li = []
        passages = document['passages']
        split_word = re.compile(r"\w+|\S")
        for r in document['relations']:
            relations.add((r['infons']['Gene1'], r['infons']['Gene2']))

        for passage in passages:
            anns = passage['annotations']
            text = passage['text']
            offset_p = passage['offset']
            index = 0
            if len(anns) == 0:
                text_li.extend(split_word.findall(text))
            else:
                anns = Annotation.sortAnns(anns)
                for ann in anns:
                    if Annotation.gettype(ann) == 'Gene':
                        for infon_type in ann['infons']:
                            if infon_type.lower() == 'ncbi gene':
                                genes.add(ann['infons'][infon_type])
                    else:
                        continue
                    for i, location in enumerate(ann['locations']):
                        offset = location['offset']
                        length = location['length']
                        text_li.extend(split_word.findall(
                            text[index:offset-offset_p]))
                        if i == len(ann['locations']) - 1:
                            ncbi_gene_id = Annotation.getNCBIID(ann)
                            text_li.append("Gene_{}".format(ncbi_gene_id))
                        index = max(offset - offset_p + length, index)
                text_li.extend(split_word.findall(text[index:]))
        return pmid, text_li, genes, relations


    def __create_examples(self, pmid, text_li, genes, relations):
        examples = []
        text_li_ori = text_li
        guids = set()
        for g1 in genes:
            for g2 in genes:
                guid = f"{pmid}_{g1}_{g2}"
                if self.testData and f"{pmid}_{g2}_{g1}" in guids:
                    continue
                text_li = text_li_ori.copy()
                if (g1, g2) in relations or (g2, g1) in relations:
                    label = "1"
                    self.label_1_count += 1
                else:
                    label = "0"
                    self.label_0_count += 1
                g1_l = "Gene_S" if g1 == g2 else "Gene_A"
                g2_l = "Gene_S" if g1 == g2 else "Gene_B"
                for i, word in enumerate(text_li):
                    if word[:5] == "Gene_":
                        if word[5:] == g1:
                            text_li[i] = g1_l
                        elif word[5:] == g2:
                            text_li[i] = g2_l
                        else:
                            text_li[i] = "Gene_N"
                text_a = " ".join(text_li)
                if self.testData:
                    guids.add(guid)
                examples.append(
                    InputExample(guid= guid, text_a=text_a, text_b=None, label=label))
        return examples

    def filterLongDocument(self, documents):
        pass

    def collate_fn(self, batch_examples):
        batch_text_a = [e.text_a for e in batch_examples]
        batch_guid = [e.guid for e in batch_examples]

        x = self.tokenizer(
            batch_text_a, return_tensors='pt', pad_to_max_length=True, max_length = 512, truncation = True)
        batch_label = torch.tensor([self.label_map[e.label] for e in batch_examples])
        return x, batch_label, batch_guid

class TriageDataSet(data.Dataset):
    label_map = {'0': 0, '1': 1}
    def __init__(self, documents, pretrained_dir):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=False)
        self.examples = []
        self.label_1_count = 0
        self.label_0_count = 0
        for i, document in enumerate(documents):
            self.examples.append(self.__create_examples(document))
        print("Examples_total_num {}.\tLabel_0_num: {}\tLabel_1_num: {}".format(
            len(self.examples), self.label_0_count, self.label_1_count))
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __create_examples(self, document):
        pmid = document['id']
        
        text_l = []
        for passage in document['passages']:
            text_l.append(passage['text'])
        text = ' '.join(text_l)

        label = '1' if document['infons']['relevant'] == 'yes' else '0'
        if label == '1':
            self.label_1_count += 1
        else:
            self.label_0_count += 1
        example = InputExample(guid=f"{pmid}", text_a=text, text_b=None, label=label)
        return example

    def collate_fn(self, batch_examples):
        batch_text_a = [e.text_a for e in batch_examples]
        batch_guid = [e.guid for e in batch_examples]

        x = self.tokenizer(
            batch_text_a, return_tensors='pt', pad_to_max_length=True, max_length = 512, truncation = True)
        batch_label = torch.tensor([self.label_map[e.label] for e in batch_examples])
        return x, batch_label, batch_guid
