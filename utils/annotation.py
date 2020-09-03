__all__ = ['Annotation']


class Annotation():

    @classmethod
    def getid(cls, ann):
        return ann['id']

    @classmethod
    def gettext(cls, ann):
        return ann['text']

    @classmethod
    def gettype(cls, ann):
        return ann['infons']['type']

    @classmethod
    def getlocations(cls, ann):
        return ann['locations']

    @classmethod
    def getNCBIID(cls, ann):
        if cls.gettype(ann) == 'Species':
            return ann['infons']['NCBI Taxonomy']
        elif cls.gettype(ann) == 'Gene':
            GENE_ID_KEYS = ['NCBI GENE', 'NCBI Gene', 'identifier']

            for key in GENE_ID_KEYS:
                rtn = ann['infons'].get(key)
                if rtn is not None:
                    break

            if rtn is None:
                raise KeyError(str(ann))
            else:
                return rtn
        else:
            raise Exception("NoSuchTypeDefined" + cls.gettype(ann))

    @classmethod
    def setNCBIID(cls, ann, id_):
        if cls.gettype(ann) == 'Species':
            ann['infons']['NCBI Taxonomy'] = str(id_)
        elif cls.gettype(ann) == 'Gene':
            GENE_ID_KEYS = ['NCBI GENE', 'NCBI Gene', 'identifier']
            ann['infons'][GENE_ID_KEYS[0]] = str(id_)
        else:
            raise Exception("NoSuchTypeDefined" + cls.gettype(ann))

    @classmethod
    def isSame(cls, ann1, ann2, checkID=True):
        checkAttr = ['text', 'locations']
        if checkID:
            checkAttr.append('NCBIID')
        for attr in checkAttr:
            if getattr(cls, 'get'+attr)(ann1) != getattr(cls, 'get'+attr)(ann2):
                return False
        return True

    @classmethod
    def sortAnns(cls, anns, duplicatedFilter=True, TBDFilter=True, geneFilter = True):
        if TBDFilter:
            anns_list = []
            for ann in anns:
                if cls.getNCBIID(ann) != 'TBD':
                    anns_list.append(ann)
            anns = anns_list
        if geneFilter:
            anns_list = []
            for ann in anns:
                if cls.gettype(ann) == 'Gene':
                    anns_list.append(ann)
            anns = anns_list
        
        anns_list = []
        for ann in anns:
            if len(ann['locations']) == 1:
                #  TODO 忽略多位置标注; Ignore multi-location annotation
                anns_list.append(ann)
        anns = anns_list

        anns = sorted(anns, key=lambda ann: (
            ann['locations'][0]['offset'], -ann['locations'][0]['length']))

        if duplicatedFilter:
            ann_set = set()
            anns_uni = []
            end_idx = -1
            for ann in anns:
                offset_ann = ann['locations'][0]['offset']
                length = ann['locations'][0]['length']
                if offset_ann < end_idx and end_idx > 0:
                    continue
                end_idx = offset_ann + length
                anns_uni.append(ann)
                # signature = (offset_ann,
                            #  ann['locations'][0]['length']
                            #  )
                # if signature not in ann_set:
                    # anns_uni.append(ann)
                # ann_set.add(signature)

            anns = anns_uni
        return anns
