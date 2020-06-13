import argparse
import time
import preprocessing
from preprocessing import Preprocessing, DependencyStructurePreprocessing, NounPhrasePreprocessing
from preprocessing import NounPreprocessing, SpacyPreprocessing
from score_bayes import score_bayes
from score_cosine import score_cosine
from score_mse import score_mse

import numpy as np
from collections import defaultdict
import os

# main body of program: DropFile
# input : input file path, root path
# output : recommended path
def dropfile(input_file: str, root_path: str, cached_DTM=None, cached_vocab=None, synonym_dict=None):
  preprocessing = Preprocessing()
  dspreprocessing = DependencyStructurePreprocessing()
  nppreprocessing = NounPhrasePreprocessing()
  npreprocessing = NounPreprocessing()
  spacypreprocessing = SpacyPreprocessing()

  ensembles = [
               # {"preprocessing": preprocessing, "scoring": score_cosine, "weight": 1},
               # {"preprocessing": dspreprocessing, "scoring": score_cosine, "weight": 1},
               # {"preprocessing": nppreprocessing, "scoring": score_cosine, "weight": 1},
               # {"preprocessing": npreprocessing, "scoring": score_cosine, "weight": 1},
               # {"preprocessing": preprocessing, "scoring": score_bayes, "weight": 1},
              {"preprocessing": spacypreprocessing, "scoring": score_cosine, "weight": 1},
               # {"preprocessing": preprocessing, "scoring": score_mse, "weight": 1},
              ]

  label_scores = []
  DTMs = []
  vocabs = []
  for i, method in enumerate(ensembles):
    if (cached_DTM is not None) and (cached_vocab is not None) and (synonym_dict is not None):
      dir_list, label_score, DTM, vocab = \
        method['scoring'](input_file, root_path, method['preprocessing'], cached_DTM[i], cached_vocab[i], synonym_dict)
    else:
      dir_list, label_score, DTM, vocab = \
        method['scoring'](input_file, root_path, method['preprocessing'], None, None, None)
    label_scores.append(label_score)
    DTMs.append(DTM)
    vocabs.append(vocab)

  score_arr = np.array(label_scores)
  print(score_arr)
  final_label_score = np.array([0.0] * score_arr.shape[1])
  for i in range(score_arr.shape[0]):
    final_label_score += score_arr[i]*ensembles[i]["weight"]

  dir_path = dir_list[final_label_score.argmax()]
  # dir_path = dir_list[label_score.index(max(label_score))]
  # print(dir_path)
  return dir_path, DTMs, vocabs



# main execution command
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='dropFile program')
  parser.add_argument('-r', '--root-path', help='root path that input file should be classified into',
                      type=str, action='store', default=os.path.join('.', 'test'))
  parser.add_argument('-i', '--input-file', help='input file initial path',
                      type=str, action='store')
  args = parser.parse_args()
  print('root path : {}, input file: {}'.format(args.root_path, args.input_file))
  if (args.input_file is None):
    parser.error("--input-file(-i) format should be specified")
  
  print("Running DropFile...")
  start = time.time()
  recommendation_path = dropfile(args.input_file, args.root_path)
  print("elapsed time: {}sec".format(time.time()-start))
  print("Execution Result: {}".format(recommendation_path))