# preprocessing code
import os
import re
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
from nltk import pos_tag
from nltk.tag import StanfordPOSTagger
from nltk.tag.stanford import StanfordNERTagger
import spacy
import time


lm = WordNetLemmatizer()
fd = FreqDist([word.lower() for word in brown.words()])
common_word_list = [t[0] for t in fd.most_common(1000)]

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

# spacy 사용한 preprocessing
# nlp : spacy 언어모델
def text2tokSpacy(text: str, nlp):
  stops = stopwords.words('english')
  texts = sent_tokenize(text)
  words = []
  # dep_list = ['ROOT','dobj','nsubj','nsubjpass','pobj','compound']
  pos_list = ['VERB','NOUN','PROPN']
  for text in texts:
    doc = nlp(text)
    for token in doc:
      if ((token.dep_ == 'ROOT') or (token.head.dep_ == 'ROOT')) and token.pos_ in pos_list and token.text.isalpha() and len(token.text) > 2:
        words.append(token.lemma_.lower())
  words = [word for word in words if word not in stops] # remove stopwords
  words = [word for word in words if word not in common_word_list]
  return words

# def build_vocab(doc_list: list):
#   vocab = {}
#   idx = 0
#   for doc in doc_list:
#       for token in doc:
#           if not token in vocab.keys():
#               vocab[token] = idx
#               idx += 1
#   print("len vocab is : ", len(vocab))
#   return vocab, synonym_dict

# root path로부터 vocabulary를 만들기 위한 함수 (file2tok, build_vocab)
def file2tok(file_path: str, nlp):
  txt = file2text(file_path)
  tok = text2tokSpacy(txt, nlp)
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
def build_DTMvec(file_path: str, vocab: dict, synonym_dict, nlp):
  doc = file2tok(file_path, nlp)
  bow = build_BoW(doc, vocab, synonym_dict)
  return bow


if __name__ == "__main__":
  test_file_path = "C\\Users\\us419\\Desktop\\NLP\\dropFile\\test\\sp\\01-overview.pdf" # change it for your own test file
  file_list = ["C:\\Users\\us419\\Desktop\\NLP\\dropFile\\test\\sp\\02-bits-ints.pdf",
  "\\Users\\us419\\Desktop\\NLP\\dropFile\\test\\sp\\03-float.pdf"]
  doc_list = list()

  start = time.time()
  nlp = spacy.load("en_core_web_sm")
  print("nlp load time: {}sec".format(time.time()-start))
  for file in file_list:
    doc_list.append(file2tok(file, nlp))
  vocab, synonym_dict = build_vocab(doc_list)
  print(vocab)
  
  # preprocessing : build DTM of files under root_path
  DTM = build_DTM(doc_list, vocab, synonym_dict)
  DTM = np.array(DTM)
  print(DTM)
  print(DTM.shape)
  
  # preprocessing : build DTMvec from input file
  dtm_vec = build_DTMvec(test_file_path, vocab, synonym_dict)
  print(dtm_vec)
  print(len(dtm_vec))

  print(synonym_dict)