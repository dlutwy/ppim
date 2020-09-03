<!-- English | [简体中文](./README.zh.md) -->
# PPIm extraction 
The implementation of method in our paper:
Extracting Protein-Protein Interactions Affected by Mutations via Auxiliary Task and Domain Pre-trained Model
## Prepareration
### requirements
requirements are listed in `requirements.txt`
### DataSet & Evaluation
  Original dataset and evaluation scripts can be downloaded [here][bc6pm],

  And some of annotation of genes are modified as shown in <a href="#modi">the end of the document</a> 
  
  The modified datasets and the results obtained by GNP are available [here][dataset_googledrive].
### Pre-trained Model
  BioBERT-Base v1.1 (+ PubMed 1M) is available [here][biobert]
 
  Original BERT is also known as bert-base-uncased.
### Eutil
  The gene normalization module(*uninvolved to our paper*) referes to the method proposed by Tung Tran in the [paper][workshop] titled *Exploring a Deep Learning Pipeline for the BioCreative VI Precision Medicine Task*. 

  This method requires the use of [eutils pacakage][eutil] and this package will automatically throttle requests according to NCBI guidelines (3 or 10 requests/second without or with an API key, respectively).
  
  Your private API key and email are supposed to be added to `./.env` if you need higher throughoutput.

## Reproducing Results
### 10-fold cross validation on train
1. RC-Only with original BERT: ` bash scripts/train_RC-BaseBERT.sh`
2. RC-Only: ` bash scripts/train_RC.sh`
3. RC+NER: ` bash scripts/train_RC+NER.sh`
4. RC+Triage: ` bash scripts/train_RC+Triage.sh`

### Train and eval on test
1. RC-Only: ` bash scripts/test_RC.sh`
2. RC+NER: ` bash scripts/test_RC+NER.sh`
3. RC+Triage: ` bash scripts/test_RC+Triage.sh`

The result of RC with confidence will be saved as `./outputjson/{loggercomment}_{epoch}.json`.

### Analysis
1. `analysis.ipynb`
   
    Predicted relations need to be post-processed here before homolo eval and exact eval.

    Scripts about case study can be found here. 
2. `head_view_bert.ipynb`: Visulalization of Attention in BERT via [BertViz][bertviz]
3. `python cross_fold_metrics.py > metrics.tsv`
    print results of the cross fold validation to `metrics.tsv`

## <span id='modi'/> Modification of Dataset
### 1. Location Error
`PMID`: `9488135`
```
{
  "text": "receptor R2", 
  "infons": {
    "type": "Gene", 
    "NCBI GENE": "7133"
  }, 
  "locations": [
    {
      "length": 11, 
      "offset": 339 -> 328
    }
  ]
}
```
### 2. Boundary Error

`PMID`: `21751375`
```
{
  "text": "CBP/b",  -> CBP
  "infons": {
    "type": "Gene", 
    "NCBI GENE": "1387"
  }, 
  "locations": [
    {
      "length": 5, -> 3
      "offset": 340 
    }
  ]
}
```
`PMID` `22014570`
```
{
  "text": "FoxM1-b",  -> FoxM1
  "infons": {
    "type": "Gene", 
    "NCBI GENE": "2305"
  },
  "locations": [
    {
      "length": 7,  -> 5
      "offset": 694
    }
  ]
}

{
  "text": "FoxM1-b", ->FoxM1
  "infons": {
    "type": "Gene", 
    "NCBI GENE": "2305"
  }, 
  "locations": [
    {
      "length": 7, -> 5 
      "offset": 801
    }
  ]
}
```
[bc6pm]: https://github.com/ncbi-nlp/BC6PM 
[dataset_googledrive]: https://drive.google.com/file/d/17MCutWfCWA2rKpPnFp6gEJATdh-IYkZX/view?usp=sharing 
[workshop]: https://biocreative.bioinformatics.udel.edu/resources/publications/bcvi-proceedings/
[eutil]: https://pypi.org/project/eutils/
[biobert]: https://github.com/dmis-lab/biobert
[bertviz]: https://github.com/jessevig/bertviz
