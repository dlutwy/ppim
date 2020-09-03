from collections import namedtuple
import pickle
import os
import sys
class HomoloQueryService():
    def __init__(self):
        # super().__init__()
        Record = namedtuple("Record", ['homoloID', 'TaxonomyID', 'geneID', 'geneSymbol', 'proteinID', 'proteinRefSeq'])
        self.homoloID2Genes = {}
        self.gene2HomoloID = {}
        n = 0
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"homologene.data")
        # print('-'*30, file = sys.stderr)
        # print("Homolo Init Data File:", path, file = sys.stderr)
        with open(path) as f:
            for line in f:
                record = Record(*line.strip().split('\t'))
                if self.homoloID2Genes.get(record.homoloID) is None:
                    self.homoloID2Genes[record.homoloID] = []
                self.homoloID2Genes[record.homoloID].append(record.geneID)
                self.gene2HomoloID[record.geneID] = record.homoloID
                n += 1
        # print('homolo num:', len(self.homoloID2Genes.keys()),'\tGenes num:' ,n, file = sys.stderr)
        # print('-'*30, file = sys.stderr)
    def getHomolo(self, geneID):
        return self.gene2HomoloID.get(geneID, "NotFound:"+geneID)

    def isHomolo(self, geneID, geneID2):
        homo1 = self.getHomo(geneID)
        homo2 = self.getHomo(geneID2)
        return homo1 == homo2