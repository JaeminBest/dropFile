import argparse
import time
from preprocessing.preprocessing import Preprocessing, DependencyStructurePreprocessing, NounPhrasePreprocessing
from preprocessing.preprocessing import NounPreprocessing, SpacyPreprocessing
from score.score_bayes import score_bayes
from score.score_cosine import score_cosine
from score.score_mse import score_mse

import numpy as np
from collections import defaultdict
import os

# main body of program: DropFile
# input : input file path, root path
# output : recommended path
def dropfile(input_file: str, root_path: str, preprocessing=None, scoring=None, cached_DTM=None, cached_vocab=None, cached_synonym_dict=None):
  normalpreprocessing = Preprocessing()
  dspreprocessing = DependencyStructurePreprocessing()
  nppreprocessing = NounPhrasePreprocessing()
  npreprocessing = NounPreprocessing()
  spacypreprocessing = SpacyPreprocessing()
  preprocessing_dict = {"Preprocessing": normalpreprocessing,
                        "DependencyStructurePreprocessing": dspreprocessing,
                        "NounPhrasePreprocessing": nppreprocessing,
                        "NounPreprocessing": npreprocessing,
                        "SpacyPreprocessing": spacypreprocessing}
  scoring_dict = {"score_mse": score_mse,
                  "score_cosine": score_cosine,
                  "score_bayes": score_bayes}
  

  if preprocessing is not None and scoring is not None:
    preprocessing_list = ["Preprocessing", "DependencyStructurePreprocessing", "NounPhrasePreprocessing",
                          "NounPreprocessing", "SpacyPreprocessing", "TargetWordChunkingPreprocessing"]
    if preprocessing not in preprocessing_list:
      print("Enter the valid preprocessing name")
      return

    if preprocessing in cached_DTM and preprocessing in cached_vocab and preprocessing in cached_synonym_dict:
      dir_list, label_score, _, _, _ = \
        scoring_dict[scoring](input_file, root_path, preprocessing_dict[preprocessing],
        cached_DTM[preprocessing], cached_vocab[preprocessing],
        cached_synonym_dict[preprocessing])
    else:
      dir_list, label_score, _, _, _ = \
          scoring_dict[scoring](input_file, root_path, preprocessing_dict[preprocessing], None, None, None)
    print(f"{preprocessing}, {scoring.__name__.split('_')[1]} score is {label_score}")

    score_arr = np.array(label_score)

    dir_path = dir_list[score_arr.argmax()]
    # dir_path = dir_list[label_score.index(max(label_score))]
    # print(dir_path)
    return dir_path, cached_DTM, cached_vocab, cached_synonym_dict

  ensembles = [
               {"preprocessing": "Preprocessing", "scoring": score_cosine, "weight": 1},
               {"preprocessing": "DependencyStructurePreprocessing", "scoring": score_cosine, "weight": 1},
               {"preprocessing": "NounPhrasePreprocessing", "scoring": score_cosine, "weight": 1},
               {"preprocessing": "NounPreprocessing", "scoring": score_cosine, "weight": 1},
               {"preprocessing": "Preprocessing", "scoring": score_bayes, "weight": 1},
               {"preprocessing": "SpacyPreprocessing", "scoring": score_cosine, "weight": 1},
               {"preprocessing": "Preprocessing", "scoring": score_mse, "weight": 1},
              ]

  label_scores = []
  if cached_DTM is None:
    cached_DTM = dict()
  if cached_vocab is None:
    cached_vocab = dict()
  if cached_synonym_dict is None:
    cached_synonym_dict = dict()

  for i, method in enumerate(ensembles):
    preprocessing_name = method["preprocessing"]
    if preprocessing_name in cached_DTM and preprocessing_name in cached_vocab \
            and preprocessing_name in cached_synonym_dict:
      dir_list, label_score, DTM, vocab, synonym_dict = \
        method['scoring'](input_file, root_path, preprocessing_dict[preprocessing_name],
                          cached_DTM[preprocessing_name], cached_vocab[preprocessing_name],
                          cached_synonym_dict[preprocessing_name])
    else:
      dir_list, label_score, DTM, vocab, synonym_dict= \
        method['scoring'](input_file, root_path, preprocessing_dict[preprocessing_name], None, None, None)
      cached_DTM[preprocessing_name] = DTM
      cached_vocab[preprocessing_name] = vocab
      cached_synonym_dict[preprocessing_name] = synonym_dict

    label_scores.append(label_score)

  score_arr = np.array(label_scores)
  print(score_arr)
  final_label_score = np.array([0.0] * score_arr.shape[1])
  for i in range(score_arr.shape[0]):
    final_label_score += score_arr[i]*ensembles[i]["weight"]

  dir_path = dir_list[final_label_score.argmax()]
  # dir_path = dir_list[label_score.index(max(label_score))]
  # print(dir_path)
  return dir_path, cached_DTM, cached_vocab, cached_synonym_dict

def prepare_env(root_path: str):

  normalpreprocessing = Preprocessing()
  dspreprocessing = DependencyStructurePreprocessing()
  nppreprocessing = NounPhrasePreprocessing()
  npreprocessing = NounPreprocessing()
  spacypreprocessing = SpacyPreprocessing()
  preprocessing_dict = {"Preprocessing": normalpreprocessing,
                        "DependencyStructurePreprocessing": dspreprocessing,
                        "NounPhrasePreprocessing": nppreprocessing,
                        "NounPreprocessing": npreprocessing,
                        "SpacyPreprocessing": spacypreprocessing}

  DTM_dict = dict()
  vocab_dict = dict()
  synonym_dict_dict = dict()

  for name, preprocessing in preprocessing_dict.items():
    # preprocessing : lookup hierarchy of root path
    directory_dict = defaultdict(list)  # empty dictionary for lookup_directory function
    dir_hierarchy = preprocessing.lookup_directory(root_path, directory_dict)  # change it to have 2 parameter

    file_list = list()
    dir_list = list()
    label_num = 0
    for tar_dir in dir_hierarchy:
      file_list += dir_hierarchy[tar_dir]
      dir_list.append(tar_dir)
      label_num += 1

    # preprocessing : build vocabulary from file_list
    # if (DTM is None) and (vocab is None) and (synonym_dict is None):
    doc_list = list()
    for file in file_list:
      doc_list.append(preprocessing.file2tok(file))
    vocab, synonym_dict = preprocessing.build_vocab(doc_list)
    # preprocessing : build DTM of files under root_path
    DTM = preprocessing.build_DTM(doc_list, vocab, synonym_dict)

    DTM_dict[name] = DTM
    vocab_dict[name] = vocab
    synonym_dict_dict[name] = synonym_dict

  return DTM_dict, vocab_dict, synonym_dict_dict


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
  # D,V,S = prepare_env('', args.root_path)
  recommendation_path = dropfile(args.input_file, args.root_path)
  print("elapsed time: {}sec".format(time.time()-start))
  print("Execution Result: {}".format(recommendation_path))