# DropFile (CS372 NLP Term Project)

### Introduction
When user download the new file, OS recommend the path to 'Downloads' directory or currently used directory. 
We want to automatically recommend the correct path by analyzing the contents of downloaded file.
That is our 'Dropfile' system.

### Quick Start

```bash
$ mkdir test (dropFile will classify it as one of the subdirectories of the "test".)
$ pip install -r requirement.txt
$ python -m spacy download en_core_web_lg/sm
$ python dropfile.py -r (root_path: default ./test) -i file_path
$ python evaluation.py -r (root_path: ./test/case(1/2/3/4/5/6) -a (preprocessing method) -b (score metric)
```
*./test* manually pre-classified by person should be located.
Due to capacity issues, pre-classified files is not posted on this repository.
&nbsp;

### How to Evaluate

With evaluation.py, you can easily measure accuracy of classification.  
If you execute `python evaluation.py -r (root_path: ./test/case(1/2/3/4/5/6) -a (preprocessing method) -b (score metric)`,
it creates and store two types of figures: First one is confusion matrix which shows the correct and wrong classifications,
and seconds are bar graphs of label score of each document. The bar graphs are generated separately for all documents.  
The all figures are stored in same location of `evaluation.py`
&nbsp;

### Environments
All required packages are in requirement.txt.
We recommend to use Python 3.7.4

### Code Structure
- dropfile.py : Recommend directory path of single file
    - function dropfile() : Directory path recommendation
    - function prepare_env() : After pre-calculation, return DTM, vocab, synonym_dict.
                               They can be used in dropfile(),
                               It is used for the purpose of improving performance by caching intermediate calculations.
- evaluation.py : Measure the accuracy using K-fold validation method.
- preprocessing
    - preprocessing.py : Extract features of each file.
        - class Preprocessing : Tokenized document into words, and lemmatize. Simplest version.
        - class DependencyParserPreprocessing : Extract only heads of all subtrees of dependency tree.
        - class NounPhrasePreprocessing : Extract only heads of noun phrase.
        - class NounPreprocessing : Extract only nouns.
        - class SpacyPreprocessing : Extract roots and children of roots of dependency trees.
        - class CFGPreprocessing : Extract noun phrases near target word, which means important words.
        - class TargetWordChunkingPreprocessing : Traverse grammar tree, and extract important words.
- score
    - score_cosine.py : calculate score using cosine similarity.
    - score_mse.py : calculate score using mse.
    - score_bayes.py : calculate score using naive bayes classifier.
    - score_CFG.py : calculate score using CFG. (variation of cosine similarity)
- tests : Directory contains subdirectories of pre-classified files, which would be used as train set
(default value of root_path)  
&nbsp;  

