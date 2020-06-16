# preprocessing code
import os
import re
from pdfminer.high_level import extract_text
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
from nltk import CFG
from nltk import RecursiveDescentParser
import spacy
import pickle
import time

print("Loading spacy!")
nlp = spacy.load('en_core_web_lg')



lm = WordNetLemmatizer()

class Preprocessing():
  def __init__(self):
    if 'DROPFILE_LOGLEVEL' in os.environ:
      self.verbose = int(os.environ['DROPFILE_LOGLEVEL'])
    else:
      self.verbose = False

  # function : lookup directory hierarchy under root_path
  # input : root_path, empty dictionary for storage
  # output : dictionary,
  #          key = 분류할 디렉토리의 절대경로,
  #          value = 그(key) 디렉토리 안에 있는 file_name(절대경로)의 리스트
  # output : dictionary that having absolute directory path as key
  #          and list of file name inside the directory as value
  # implementation : os 라이브러리 사용하면 될듯, UNIX 환경 가정 (mac, ubuntu)
  def lookup_directory(self, root_path: str, directory_dict: dict):
    try:
      root = os.listdir(root_path)
      if root:
        for filename in root:
          full_filename = os.path.join(root_path, filename)
          if os.path.isdir(full_filename):
            self.lookup_directory(full_filename, directory_dict)
          else:
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
  def file2text(self, file_path: str):
    with open(file_path, 'rb') as f:
      text = pickle.load(f) # 이제 모든 강의파일은 pickle로 저장되있음 (not pdf)

    # start = time.time()
    # text = extract_text(file_path) # extract text from pdf file
    # if self.verbose:
    #     print(f"raw text : {text[:100]}")
    #     print(f"extract_text takes {time.time()-start:.4f} s.")
    return text

  # function : convert text into list of parsed token
  # input : text (불용어, 문장부호, 띄어쓰기 등 포함)
  # output : list of parsed token, 의미를 가지는 토큰만 포함한 리스트
  # implementation : Regex 라이브러리로 필터링
  def text2tok(self, text: str):
    words = word_tokenize(text) # tokenize words by nltk word_toknizer
    stops = stopwords.words('english')
    start = time.time()
    words = [word.lower() for word in words] # convert uppercase to lowercase
    words = [word for word in words if re.match('^[a-zA-Z]\w+$', word)] # regex version of above line
    words = [lm.lemmatize(word) for word in words] # lemmatize words
    # words = [word for word in words if word not in common_word_list] # exclude common words in corpus
    if self.verbose:
        print(f"After tokenize : {words[:30]}")
        print(f"text2tok takes {time.time()-start:.4f} s.")
    return words

  # root path로부터 vocabulary를 만들기 위한 함수 (file2tok, build_vocab)
  def file2tok(self, file_path: str):
    txt = self.file2text(file_path)
    tok = self.text2tok(txt)
    return tok

  def build_vocab(self, doc_list: list):
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
  def build_BoW(self, doc: list, vocab: dict, synonym_dict: dict):
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
  def build_DTM(self, doc_list: list, vocab: dict, synonym_dict):
    DTM = []
    for doc in doc_list:
      DTM.append(self.build_BoW(doc, vocab, synonym_dict))
    return DTM


  # function : make DTMvec from input_file (bow of input_file)
  # input : file_path of input_file, vocab
  # output : DTMvec (bow)
  def build_DTMvec(self, file_path: str, vocab: dict, synonym_dict):
    txt = self.file2text(file_path)
    doc = self.text2tok(txt)
    bow = self.build_BoW(doc, vocab, synonym_dict)
    return bow

class DependencyStructurePreprocessing(Preprocessing):
  def text2tok(self, text: str):
    doc = nlp(text)
    head_list = []
    for token in doc:
      if [child for child in token.children] != []:
        head_list.append(token.lemma_)
    words = [word.lower() for word in head_list]
    words = [word for word in words if re.match('^[a-zA-Z]\w+$', word)]  # regex version of above line
    if self.verbose:
        print(f"After tokenize : {words[:30]}")
    return words

class NounPhrasePreprocessing(Preprocessing):
  def text2tok(self, text: str):
    doc = nlp(text)
    head_list = [chunk.root.lemma_ for chunk in doc.noun_chunks]
    words = [word.lower() for word in head_list]
    words = [word for word in words if re.match('^[a-zA-Z]\w+$', word)]  # regex version of above line
    if self.verbose:
        print(f"After tokenize : {words[:30]}")
    return words


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
KEYWORD_LIST = ['learn', 'study', 'represent', 'summary', 'structure', 'application', 'apply', 'involve', 'base',
                  'require', 'achieve', 'course', 'determine', 'property', 'develop', 'create']

class NounPreprocessing(Preprocessing):
  # function : convert text into list of parsed token
  # input : text (불용어, 문장부호, 띄어쓰기 등 포함)
  # output : list of parsed token, 의미를 가지는 토큰만 포함한 리스트
  # implementation : Regex 라이브러리로 필터링
  def text2tok(self, text: str):
    words = word_tokenize(text) # tokenize words by nltk word_toknizer
    words_with_pos = pos_tag(words)
    noun = ["NN", "NNS", "NNP", "NNPS"]
    words_with_pos = [word[0] for word in words_with_pos if word[1] in noun]
    words = [word.lower() for word in words_with_pos] # convert uppercase to lowercase
    words = [word for word in words if re.match('^[a-zA-Z]\w+$', word)] # regex version of above line
    words = [lm.lemmatize(word) for word in words] # lemmatize words
    # print(common_word_list)
    if self.verbose:
        print(f"After tokenize : {words[:30]}")
    return words

class SpacyPreprocessing(Preprocessing):
  # spacy 사용한 preprocessing
  # nlp : spacy 언어모델
  def text2tok(self, text: str):
    stops = stopwords.words('english')
    texts = sent_tokenize(text)
    words = []
    # dep_list = ['ROOT','dobj','nsubj','nsubjpass','pobj','compound']
    pos_list = ['VERB', 'NOUN', 'PROPN']
    for text in texts:
      doc = nlp(text)
      for token in doc:
        if ((token.dep_ == 'ROOT') or (
                token.head.dep_ == 'ROOT')) and token.pos_ in pos_list and token.text.isalpha() and len(token.text) > 2:
          words.append(token.lemma_.lower())
    words = [word for word in words if word not in stops]  # remove stopwords
    if self.verbose:
        print(f"After tokenize : {words[:30]}")
    return words

  # function : make DTMvec from input_file (bow of input_file)
  # input : file_path of input_file, vocab
  # output : DTMvec (bow)
  def build_DTMvec(self, file_path: str, vocab: dict, synonym_dict):
    doc = self.file2tok(file_path)
    bow = self.build_BoW(doc, vocab, synonym_dict)
    return bow

class TargetWordChunkingPreprocessing(Preprocessing):
  # spacy 사용한 preprocessing
  # nlp : spacy 언어모델
  # target word : 강의의 요점이나 핵심이 담길 만한 단어
  # method : target word가 등장한 문장, 그 다음 두 문장까지 문장에서 Noun phrase를 추출하여 DTM 구성
  def text2tok(self, text: str):
    target_word = ["purpose", "object", "goal", "objective", "aim", "learn", "study", "overview"]
    grammar = CFG.fromstring("""
    S -> NP VP
    VP -> V NP | V NP PP
    PP -> P NP
    V -> learn | study
    Det -> "a" | "an" | "the" | "our"
    N -> "goal" | "purpose" | "object" | "overview"
    P -> "in" | "on" | "at" | "by" | "with" | "about"
    """)
    rd_parser = RecursiveDescentParser(grammar)
    chunk_list = []
    doc = nlp(text)
    flag = 0
    for sent in doc.sents:
      if flag > 0:
        for chunk in sent.noun_chunks:
          chunk_list.append(chunk.lemma_)
          flag -= 1
      for token in sent:
        if token.lemma_ in target_word:
          flag = 2
          for chunk in sent.noun_chunks:
            chunk_list.append(chunk.lemma_)

    words = [word.lower() for word in chunk_list]
    words = [word for word in words if re.match('^[a-zA-Z]\w+$', word)]
    return words

class CFGPreprocessing(Preprocessing):
  def __init__(self):
    if 'DROPFILE_LOGLEVEL' in os.environ:
      self.verbose = int(os.environ['DROPFILE_LOGLEVEL'])
    else:
      self.verbose = False

  def text2tok(self, text: str):
    tagged_sents = []
    text = text.lower()
    doc = nlp(text)
    for sent in doc.sents:
      words = [token for token in sent if re.match('^[a-zA-Z]\w+$', token.text) or token.text == ',']
      if words:
        tagged_sents.append(words)

    return tagged_sents

  # function : make DTMvec from input_file (bow of input_file)
  # input : file_path of input_file, vocab
  # output : DTMvec (bow)
  def build_DTMvec(self, file_path: str, vocab: dict, synonym_dict):
    # txt = file2text(file_path)
    # doc = text2tok(txt)
    doc = self.extract_mean(file_path)
    bow = self.build_BoW(doc, vocab, synonym_dict)
    return bow

  # convert token into preferarable way
  def convert_token(self, tokens):
    pos_sent = list()
    index_list = list()
    for i, tok in enumerate(tokens):
      temp = self.convert_single_tok(tok.text, tok.pos_, tok.tag_)
      if temp == 'UNK':
        continue
      pos_sent.append(temp)
      index_list.append(i)
    return pos_sent, index_list


  # convert single token into preferarable way
  # only used tag is "VERB", "VBN", "NOUN", "IN", "COMMA", "THAT", "BE', ""UNK"
  # My tag is "VERB", "NOUN", "THAT", "BE", "UNK", "COMMA"
  def convert_single_tok(self, word, pos, tag):
    if word == 'that':
      rv = 'THAT'
    elif word == ',':
      rv = 'COMMA'
    elif pos == 'VERB':
      rv = 'VERB'
    elif lm.lemmatize(word, "v") == "be":
      rv = "BE"
    elif pos == 'PRON' or tag.startswith("PR"):
      rv = 'NOUN'
    elif tag.startswith("NN"):
      rv = 'NOUN'
    elif tag.startswith("VB"):
      rv = "VERB"
    else:
      rv = 'UNK'
    return rv


  def nltkTree_traverse_index(self, t, idx):
    try:
      label = t.label()
    except AttributeError:
      return "{}/{} ".format(t, idx), idx + 1
    else:
      string = "({} ".format(label)
      for child in t:
        nstring, new_idx = self.nltkTree_traverse_index(child, idx)
        string += nstring
        idx = new_idx
      string += ") "
      return string, idx


  # get tagged sentence and generate CFG. It returns meaningful words in sentence.
  def generate_cfg(self, token_sent):
    pos_sent, index_list = self.convert_token(token_sent)
    tok_list = [token_sent[idx] for idx in index_list]

    tree_list = list()

    for tree in rd_parser.parse(pos_sent):
      # print(tree)
      tree_list.append(tree)

    answer = list()

    # scenario 0 : there is matching grammar
    mean_words = []
    for tree in tree_list:
      nstring, _ = self.nltkTree_traverse_index(tree, 0)
      ntree = nltk.Tree.fromstring(nstring)
      ntree = nltk.tree.ParentedTree.convert(ntree)
      # print("ntree: ",ntree)
      # print("subtree: ",ntree.subtrees())
      # target_verb,target_obj,target_subj,REVERSE_FLAG = nltkTree_VP_search(ntree.subtrees(),tok_list)
      mean_words = self.tree_extract(ntree.subtrees(), tok_list)
      # print('temp mean_words : ',mean_words)
      if mean_words:
        break

    return mean_words


  # convert leaves to words in tok_list
  def leaves2words(self, leaves, tok_list):
    words = []
    for leaf in leaves:
      tag, idx = self.get_tag_idx(leaf)

      word = tok_list[idx].text
      words.append(word)
    return words


  #
  # extract meaningful words from CFG parsed tree
  def tree_extract(self, subtrees, tok_list):
    for tree in subtrees:
      try:
        tree.label()
      except AttributeError:
        continue
      else:
        if tree.label() != 'S':
          leaves = tree.leaves()
          words = self.leaves2words(leaves, tok_list)

          if self.contain_list(words):
            return words
        else:
          continue

        # print('label :',tree.label())
        # print('leaves :', tree.leaves())


  # check whether words list has target word
  def contain_list(self, words):
    for word in words:
      lemma = lm.lemmatize(word)
      if lemma in KEYWORD_LIST:
        return True
    return False


  # get tag and idx of leaf
  def get_tag_idx(self, word):
    tag, idx = word.split('/')
    return tag, int(idx)


  # lemmatize given token. if token pos is VERB, it lemmatize it with 'v' option.
  def lemmatize(self, token):
    if token.pos_ == "VERB":
      return lm.lemmatize(token.text, 'v')
    else:
      return lm.lemmatize(token.text)


  def is_contain(self, token_sent):
    for target in KEYWORD_LIST:
      if target in list(map(lambda x: self.lemmatize(x), token_sent)):
        return True
    return False


  def filter_target(self, token_sents):
    filter_sents = []
    for sent in token_sents:
      if self.is_contain(sent):
        filter_sents.append(sent)
    return filter_sents


  # extract meaningful words from input file path
  def extract_mean(self, file):
    token_sent = self.file2tok(file)
    filter_sent = self.filter_target(token_sent)
    # print('filter sent:',filter_sent)
    mean_words = []
    for sent in filter_sent:
      temp = self.generate_cfg(sent)
      if temp:
        mean_words.extend(temp)

    # remove stopwords and comma
    stops = stopwords.words('english')
    mean_words = [word for word in mean_words if word != ',']
    mean_words = [word for word in mean_words if word not in stops]

    return mean_words

if __name__ == "__main__":
  preprocessing = NounPhrasePreprocessing()

  # change it for your own test file
  for subject in ['A', "A'", "A''", "B", "B'", "B''", "C", "C'", "C''"]:
    for dir_name, _, file_names in os.walk(os.path.join('textfiles', subject)):
      test_file_path = [os.path.join(dir_name, file_name) for file_name in file_names]

    tokens = []
    for file in test_file_path:
      tokens += preprocessing.file2tok(file)

    print(FreqDist(tokens).most_common(10))

  quit()

  file_list = ["C://Users//us419//Desktop//OS//01_introduction.pdf",
  "C://Users//us419//Desktop//OS//02_kernel.pdf",
  "C://Users//us419//Desktop//OS//03_scheduling.pdf"]
  doc_list = list()
  for file in file_list:
    doc_list.append(preprocessing.file2tok(file))
  vocab, synonym_dict = preprocessing.build_vocab(doc_list)
  print(vocab)
  
  # preprocessing : build DTM of files under root_path
  DTM = preprocessing.build_DTM(doc_list, vocab, synonym_dict)
  DTM = np.array(DTM)
  print(DTM)
  print(DTM.shape)
  
  # preprocessing : build DTMvec from input file
  dtm_vec = preprocessing.build_DTMvec(test_file_path, vocab, synonym_dict)
  print(dtm_vec)
  print(len(dtm_vec))

  print(synonym_dict)