# preprocessing code
import os
import re
import nltk
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.high_level import extract_text
from io import StringIO
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, brown
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
import numpy as np
import multiprocessing as mp
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
from pickle import load
from spacy.lang.en import English
import spacy

lm = WordNetLemmatizer()

# function : lookup directory hierarchy under root_path
# input : root_path, empty dictionary for storage
# output : dictionary, 
#          key = 분류할 디렉토리의 절대경로, 
#          value = 그(key) 디렉토리 안에 있는 file_name(절대경로)의 리스트 
# output : dictionary that having absolute directory path as key 
#          and list of file name inside the directory as value
# implementation : os 라이브러리 사용하면 될듯, UNIX 환경 가정 (mac, ubuntu)
def lookup_directory(root_path: str, directory_dict: dict):
  try:
    root = os.listdir(root_path)
    if root:
      for filename in root:
        full_filename = os.path.join(root_path, filename)
        if os.path.isdir(full_filename):
          lookup_directory(full_filename, directory_dict)
        else:
          extension = os.path.splitext(full_filename)[-1]
          if extension == '.pdf':
            if root_path in directory_dict:
              directory_dict[root_path].append(full_filename)
            else:
              directory_dict[root_path] = [full_filename]
    else:
      return directory_dict # add return value
  except PermissionError:
    pass
  return directory_dict # add return value


# function : read pdf file and convert into text(string) format
# input : file_path
# output : text
# implementation : pdfminer 라이브러리 사용 (ref: https://lsjsj92.tistory.com/304)

def file2text(file_path: str):
  text = extract_text(file_path) # extract text from pdf file
  return text



# function : convert text into list of parsed token
# input : text (불용어, 문장부호, 띄어쓰기 등 포함)
# output : list of parsed token, 의미를 가지는 토큰만 포함한 리스트
# implementation : Regex 라이브러리로 필터링
nlp = spacy.load('en_core_web_sm')

def text2tok(text: str):
  tagged_sents = []
  text = text.lower()
  doc = nlp(text)
  for sent in doc.sents:
    words = [token for token in sent if re.match('^[a-zA-Z]\w+$', token.text) or token.text==',']
    if words:
      tagged_sents.append(words)

  return tagged_sents


# root path로부터 vocabulary를 만들기 위한 함수 (file2tok, build_vocab)
def file2tok(file_path: str):
  txt = file2text(file_path)
  tok = text2tok(txt)
  return tok 

def build_vocab(doc_list: list):
  vocab = {}
  synonym_dict = {}
  idx = 0
  for doc in doc_list:
      for token in doc:
          find = False
          for synset in wn.synsets(token):
              synonyms = synset.lemma_names()
              for synonym in synonyms:
                if synonym in vocab.keys() and token != synonym:
                  synonym_dict[synonym] = token
                  find = True
                  break
              if find:
                break
          if find:
            continue
          if not token in vocab.keys():
              vocab[token] = idx
              idx += 1
  return vocab, synonym_dict


# function : build BoW
# input : doc (list of tokens), vocab (root_path으로부터 build_vocab로 만든 vocabulary)
# output : bow 
#          ex) [1,2,4,,...]
def build_BoW(doc: list, vocab: dict, synonym_dict: dict):
  freqdist = FreqDist(doc)
  bow = [0] * len(vocab.keys())
  for token in freqdist.keys():
    try:
        if token in vocab.keys():
          bow[vocab[token]] += freqdist[token]
        else:
          bow[vocab[synonym_dict[token]]] += freqdist[token]
    except KeyError:  # token이 vocabulary에 없는 경우 지금은 pass로 구현했지만 다른 구현 고려 가능
        pass          # ex : vocab에 UNK라는 token을 만들어 주고 bow[vocab['UNK']] += 1
  return bow


# function : build vocab list out of each list_of parsed token
# input : list of doc
#        ex) [["I", "love", "her", ....](1st file),
#             ["I", "love", "her", ....](2nd file),[..],[..]]
# output : list of bow, DTM
#         ex) DTM : [[1,2,4,,...](1st file),[3,5,2,...](2nd file), ...]
def build_DTM(doc_list: list, vocab: dict, synonym_dict):
  DTM = []
  for doc in doc_list:
    DTM.append(build_BoW(doc, vocab, synonym_dict))
  return DTM


# function : make DTMvec from input_file (bow of input_file)
# input : file_path of input_file, vocab
# output : DTMvec (bow)
def build_DTMvec(file_path: str, vocab: dict, synonym_dict):
  # txt = file2text(file_path)
  # doc = text2tok(txt)
  doc = extract_mean(file_path)
  bow = build_BoW(doc, vocab, synonym_dict)
  return bow

# ===============================================================================================================================
# Below is newly added my implementation 
# ===============================================================================================================================

# Atomic grammar class : VERB, NOUN, BE, THAT, COMMA
new_grammar = r"""
S -> NP VP
NP -> N | N NP | N "THAT" VP | "COMMA" NP | N "THAT" S
VP -> V | V VP | V NP | V "THAT" S | V S
N -> "NOUN"
V -> "BE" "VERB" | "BE" | "VERB"
"""

grammar = nltk.CFG.fromstring(new_grammar)
rd_parser = nltk.RecursiveDescentParser(grammar)

# convert token into preferarable way
def convert_token(tokens):
  pos_sent = list()
  index_list = list()
  for i,tok in enumerate(tokens):
    temp = convert_single_tok(tok.text,tok.pos_,tok.tag_)
    if temp =='UNK':
      continue
    pos_sent.append(temp)
    index_list.append(i)
  return pos_sent,index_list


# convert single token into preferarable way
# only used tag is "VERB", "VBN", "NOUN", "IN", "COMMA", "THAT", "BE', ""UNK"
# My tag is "VERB", "NOUN", "THAT", "BE", "UNK", "COMMA"
def convert_single_tok(word, pos, tag):
  if word=='that':
    rv = 'THAT'
  elif word == ',':
    rv = 'COMMA'
  elif pos=='VERB':
    rv = 'VERB'
  elif lm.lemmatize(word,"v")=="be":
    rv = "BE"
  elif pos=='PRON' or tag.startswith("PR"):
    rv = 'NOUN'
  elif tag.startswith("NN"):
    rv = 'NOUN'
  elif tag.startswith("VB"):
    rv = "VERB"
  else:
    rv = 'UNK'
  return rv

def nltkTree_traverse_index(t,idx):
  try:
    label = t.label()
  except AttributeError:
    return "{}/{} ".format(t,idx),idx+1
  else:
    string = "({} ".format(label)
    for child in t:
      nstring,new_idx=nltkTree_traverse_index(child,idx)
      string += nstring
      idx = new_idx
    string+=") "
    return string, idx


# get tagged sentence and generate CFG. It returns meaningful words in sentence. 
def generate_cfg(token_sent):
  
  pos_sent, index_list = convert_token(token_sent)
  tok_list = [token_sent[idx] for idx in index_list]
  
  tree_list = list()

  for tree in rd_parser.parse(pos_sent):
    # print(tree)
    tree_list.append(tree)

  answer = list()
            
  # scenario 0 : there is matching grammar
  mean_words = []
  for tree in tree_list:
    nstring,_ = nltkTree_traverse_index(tree,0)
    ntree = nltk.Tree.fromstring(nstring)
    ntree = nltk.tree.ParentedTree.convert(ntree)
    # print("ntree: ",ntree)
    # print("subtree: ",ntree.subtrees())
    # target_verb,target_obj,target_subj,REVERSE_FLAG = nltkTree_VP_search(ntree.subtrees(),tok_list)
    mean_words = tree_extract(ntree.subtrees(), tok_list)
    # print('temp mean_words : ',mean_words)
    if mean_words:
      break

  return mean_words
  
# convert leaves to words in tok_list 
def leaves2words(leaves, tok_list):
  words = []
  for leaf in leaves:
    tag, idx = get_tag_idx(leaf)

    word = tok_list[idx].text
    words.append(word)
  return words
#   
# extract meaningful words from CFG parsed tree 
def tree_extract(subtrees, tok_list):
  for tree in subtrees:
    try:
      tree.label()
    except AttributeError:
      continue
    else:
      if tree.label() != 'S':
        leaves = tree.leaves()
        words = leaves2words(leaves,tok_list)
        
        if contain_list(words):
          return words
      else:
        continue

      # print('label :',tree.label())
      # print('leaves :', tree.leaves())


# check whether words list has target word
def contain_list(words):
  for word in words:
    lemma = lm.lemmatize(word)
    if lemma in KEYWORD_LIST:
      return True
  return False
  
# get tag and idx of leaf
def get_tag_idx(word):
  tag, idx = word.split('/')
  return tag, int(idx)
  
KEYWORD_LIST = ['learn', 'study', 'represent', 'summary', 'structure', 'application', 'apply','involve', 'base','require','achieve', 'course','determine', 'property', 'develop','create']

# lemmatize given token. if token pos is VERB, it lemmatize it with 'v' option. 
def lemmatize(token):
  if token.pos_ == "VERB":
    return lm.lemmatize(token.text,'v')
  else:
    return lm.lemmatize(token.text)

def is_contain(token_sent):
  for target in KEYWORD_LIST:
    if target in list(map(lambda x: lemmatize(x),token_sent)):
      return True
  return False

def filter_target(token_sents):
  filter_sents = []
  for sent in token_sents:
    if is_contain(sent):
      filter_sents.append(sent)
  return filter_sents

# extract meaningful words from input file path
def extract_mean(file):
  token_sent = file2tok(file)
  filter_sent = filter_target(token_sent)
  # print('filter sent:',filter_sent)
  mean_words = []
  for sent in filter_sent:
    temp = generate_cfg(sent)
    if temp:
      mean_words.extend(temp)

  # remove stopwords and comma
  stops = stopwords.words('english')
  mean_words = [word for word in mean_words if word != ',']
  mean_words = [word for word in mean_words if word not in stops]

  return mean_words
    


# execution part
if __name__ == "__main__":
  test_file_path = "C://Users//us419//Desktop//OS//04_programming_interface.pdf" # change it for your own test file
  file_list = ["./verb_check/paper1.pdf","./verb_check/02-bits-ints.pdf", "./verb_check/05ArraysStrings.pdf", "./verb_check/class02_cs230s19_Logic design.pdf", "./verb_check/lecture15_Goodrich_6e_Ch09_PriorityQueues.pdf", "./verb_check/lecture17-ch4.pdf"]
  
  doc_list = list()

  # print(token_sents)
  brown = nltk.corpus.brown.sents(categories='editorial')[5]
  text = ' '.join(brown)
  text1 = "The ClpP activators are remarkable examples of small molecules that inhibit protein-protein interactions but also result in a gain of function."
  text2 = "a rat kidney tubular cell line, metformin could stimulate AMPKα phosphorylation."
  
  # file = file_list[5]
  
  # extract_mean(file)


