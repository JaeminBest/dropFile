import argparse
import time
from preprocessing.preprocessing import Preprocessing, DependencyStructurePreprocessing, NounPhrasePreprocessing
from preprocessing.preprocessing import NounPreprocessing, SpacyPreprocessing, TargetWordChunkingPreprocessing
from preprocessing.preprocessing import CFGPreprocessing
from score.score_bayes import score_bayes
from score.score_cosine import score_cosine
from score.score_mse import score_mse
from score.score_CFG import score_CFG
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import pickle
import platform

OSTYPE = platform.system()
plot_number = 0

# main body of program: DropFile
# input : input file path, root path
# output : recommended path
def dropfile(input_file: str, root_path: str, cached_DTM=None, cached_vocab=None, cached_synonym_dict=None, verbose=True, preprocessing=None, scoring=None):
  os.environ['DROPFILE_LOGLEVEL'] = "1" if verbose else "0"
  global plot_number

  normalpreprocessing = Preprocessing()
  dspreprocessing = DependencyStructurePreprocessing()
  nppreprocessing = NounPhrasePreprocessing()
  npreprocessing = NounPreprocessing()
  spacypreprocessing = SpacyPreprocessing()
  twcpreprocessing = TargetWordChunkingPreprocessing()
  cfgpreprocessing = CFGPreprocessing()
  preprocessing_dict = {"Preprocessing": normalpreprocessing,
                        "DependencyStructurePreprocessing": dspreprocessing,
                        "NounPhrasePreprocessing": nppreprocessing,
                        "NounPreprocessing": npreprocessing,
                        "SpacyPreprocessing": spacypreprocessing,
                        "TargetWordChunkingPreprocessing": twcpreprocessing,
                        "CFGPreprocessing": cfgpreprocessing}
  scoring_dict = {"score_mse": score_mse,
                  "score_cosine": score_cosine,
                  "score_bayes": score_bayes,
                  "score_CFG": score_CFG}
  

  if preprocessing is not None and scoring is not None:
    preprocessing_list = ["Preprocessing", "DependencyStructurePreprocessing", "NounPhrasePreprocessing",
                          "NounPreprocessing", "SpacyPreprocessing", "TargetWordChunkingPreprocessing","CFGPreprocessing"]
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
    if verbose:
      print(label_score)

    score_arr = np.array(label_score).astype(float)
    score_arr = score_arr / sum(score_arr)

    dir_path = dir_list[score_arr.argmax()]

    case = os.listdir(root_path)[0]
    print(f"********** {case} store score ********")
    with open(f'MaxMinDev_{case}', 'wb') as file:  # OS dependency
      score_max = np.max(score_arr)
      score_min = np.min(score_arr)
      dev = score_max - score_min
      MaxMindict = defaultdict(list)
      if OSTYPE == "Darwin":
        MaxMindict[input_file.split("/")[-1]] = [score_max, score_min, dev]
      elif OSTYPE == "Linux":
        MaxMindict[input_file.split("/")[-1]] = [score_max, score_min, dev]
      else:
        MaxMindict[input_file.split("\\")[-1]] = [score_max, score_min, dev]
      pickle.dump(MaxMindict, file)

    plt.figure(plot_number)
    plot_number += 1
    directory_name = [path.split('/')[-1].split('\\')[-1] for path in dir_list]
    y = score_arr
    x = np.arange(len(y))
    xlabel = directory_name
    if OSTYPE == "Darwin":
      plt.title("Label Score of {}".format(input_file.split('/')[-2] + '_' + input_file.split("/")[-1]))
    elif OSTYPE == "Linux":
      plt.title("Label Score of {}".format(input_file.split('/')[-2] + '_' + input_file.split("/")[-1]))
    else:  # Windows
      plt.title(
        "Label Score of {}".format(input_file.split('\\')[-2] + '_' + input_file.split("\\")[-1]))
    plt.bar(x, y, color="blue")
    plt.xticks(x, xlabel)
    if OSTYPE == "Darwin":
      plt.savefig("label_score_{}.png".format(input_file.split('/')[-2] + '_' + input_file.split("/")[-1]))
    elif OSTYPE == "Linux":
      plt.savefig("label_score_{}.png".format(input_file.split('/')[-2] + '_' + input_file.split("/")[-1]))
    else:  # Windows
      plt.savefig("label_score_{}.png".format(input_file.split('\\')[-2] + '_' + input_file.split("\\")[-1]))
    plt.close(plot_number - 1)
    return dir_path, cached_DTM, cached_vocab, cached_synonym_dict

  ensembles = [
               {"preprocessing": "Preprocessing", "scoring": score_cosine, "weight": 1},
               {"preprocessing": "DependencyStructurePreprocessing", "scoring": score_cosine, "weight": 1},
               {"preprocessing": "NounPhrasePreprocessing", "scoring": score_cosine, "weight": 1},
               {"preprocessing": "NounPreprocessing", "scoring": score_cosine, "weight": 1},
               {"preprocessing": "Preprocessing", "scoring": score_bayes, "weight": 1},
               {"preprocessing": "SpacyPreprocessing", "scoring": score_cosine, "weight": 1},
               {"preprocessing": "Preprocessing", "scoring": score_mse, "weight": 1},
               {"preprocessing": "CFGPreprocessing", "scoring": score_CFG, "weight": 1},
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
  for i in range(score_arr.shape[0]):
    score_arr[i] = score_arr[i]/sum(score_arr[i])
  if verbose:
      print(score_arr)
  final_label_score = np.array([0.0] * score_arr.shape[1])
  for i in range(score_arr.shape[0]):
    final_label_score += score_arr[i]*ensembles[i]["weight"]

  case = os.listdir(root_path)[0]
  print(f"********** {case} store score ********")
  with open(f'MaxMinDev_{case}', 'wb') as file:  # OS dependency
    score_max = np.max(final_label_score)
    score_min = np.min(final_label_score)
    dev = score_max - score_min
    MaxMindict = defaultdict(list)
    if OSTYPE == "Darwin":
      MaxMindict[input_file.split("/")[-1]] = [score_max, score_min, dev]
    elif OSTYPE == "Linux":
      MaxMindict[input_file.split("/")[-1]] = [score_max, score_min, dev]
    else:
      MaxMindict[input_file.split("\\")[-1]] = [score_max, score_min, dev]
    pickle.dump(MaxMindict, file)

  print("Your OS is ", OSTYPE)
  plt.figure(plot_number)
  plot_number += 1
  directory_name = [path.split('/')[-1].split('\\')[-1] for path in dir_list]
  y = final_label_score
  x = np.arange(len(y))
  xlabel = directory_name
  if OSTYPE == "Darwin":
    plt.title("Label Score of {}".format(input_file.split('/')[-2] + '_' + input_file.split("/")[-1]))
  elif OSTYPE == "Linux":
    plt.title("Label Score of {}".format(input_file.split('/')[-2] + '_' + input_file.split("/")[-1]))
  else:  # Windows
    plt.title("Label Score of {}".format(input_file.split('/')[-1].split("\\")[-2] + '_' + input_file.split("\\")[-1]))

  plt.bar(x, y, color="blue")
  plt.xticks(x, xlabel)

  if OSTYPE == "Darwin":
    plt.savefig("label_score_{}.png".format(input_file.split('/')[-2] + '_' + input_file.split("/")[-1]))
  elif OSTYPE == "Linux":
    plt.savefig("label_score_{}.png".format(input_file.split('/')[-2] + '_' + input_file.split("/")[-1]))
  else:  # Windows
    plt.savefig("label_score_{}.png".format(input_file.split('/')[-1].split("\\")[-2] + '_' +
                                            input_file.split('/')[-1].split("\\")[-1]))

  plt.close(plot_number-1)
  try:
    dir_path = dir_list[final_label_score.argmax()]
  except:
    dir_path = ''
    
  return dir_path, cached_DTM, cached_vocab, cached_synonym_dict

def prepare_env(root_path: str, cached_tokens=None, verbose=False):
  os.environ['DROPFILE_LOGLEVEL'] = "1" if verbose else "0"

  normalpreprocessing = Preprocessing()
  dspreprocessing = DependencyStructurePreprocessing()
  nppreprocessing = NounPhrasePreprocessing()
  npreprocessing = NounPreprocessing()
  spacypreprocessing = SpacyPreprocessing()
  twcpreprocessing = TargetWordChunkingPreprocessing()
  cfgpreprocessing = CFGPreprocessing()
  preprocessing_dict = {"Preprocessing": normalpreprocessing,
                        "DependencyStructurePreprocessing": dspreprocessing,
                        "NounPhrasePreprocessing": nppreprocessing,
                        "NounPreprocessing": npreprocessing,
                        "SpacyPreprocessing": spacypreprocessing,
                        "TargetWordChunkingPreprocessing": twcpreprocessing,
                        "CFGPreprocessing": cfgpreprocessing}

  DTM_dict = dict()
  vocab_dict = dict()
  synonym_dict_dict = dict()

  start = time.time()
  directory_dict = defaultdict(list)  # empty dictionary for lookup_directory function
  dir_hierarchy = normalpreprocessing.lookup_directory(root_path, directory_dict)
  file_list = list()
  doc_dict = dict()

  for tar_dir in dir_hierarchy:
      file_list += dir_hierarchy[tar_dir]

  if cached_tokens is None:
    tokens_dict = defaultdict(dict)
  else:
    tokens_dict = cached_tokens

  for file in file_list:
    if file not in tokens_dict["Preprocessing"]:
      doc_dict[file] = normalpreprocessing.file2text(file)
  if verbose:
    print(f"file2text takes {time.time() - start:.4f} s.")

  for name, preprocessing in preprocessing_dict.items():
    if verbose:
      print(f"{name} started")
    # preprocessing : lookup hierarchy of root path
    directory_dict = defaultdict(list)  # empty dictionary for lookup_directory function

    start = time.time()
    dir_hierarchy = preprocessing.lookup_directory(root_path, directory_dict)  # change it to have 2 parameter
    if verbose:
        print(f"{name}.lookup_directory takes {time.time()-start:.4f} s.")

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
    start = time.time()
    for file in file_list:
      if name in tokens_dict and file in tokens_dict[name]:
        tokens = tokens_dict[name][file]
      else:
        tokens = preprocessing.text2tok(doc_dict[file])
      doc_list.append(tokens)
      tokens_dict[name][file] = tokens

    if verbose:
        print(f"{name}.text2tok takes {time.time()-start:.4f} s.")
    start = time.time()
    vocab, synonym_dict = preprocessing.build_vocab(doc_list)
    if verbose:
        print(f"{name}.build_vocab takes {time.time()-start:.4f} s.")
    # preprocessing : build DTM of files under root_path
    start = time.time()
    DTM = preprocessing.build_DTM(doc_list, vocab, synonym_dict)
    if verbose:
        print(f"{name}.build_DTM takes {time.time()-start:.4f} s.")

    DTM_dict[name] = DTM
    vocab_dict[name] = vocab
    synonym_dict_dict[name] = synonym_dict

  return DTM_dict, vocab_dict, synonym_dict_dict, tokens_dict


# main execution command
if __name__=='__main__':
  import shutil
  import os

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
  print("prepare_env...")
  start = time.time()
  # D,V,S,T = prepare_env(args.root_path, verbose=True)
  print("elapsed time: {}sec".format(time.time() - start))
  shutil.copy("C:\\dropFile\\textfiles\\A''\\01Intro.pdf", "C:\\dropFile\\test\\A")
  print("update with a new file...")
  start = time.time()
  # D,V,S,T = prepare_env(args.root_path, T, verbose=False)
  print("elapsed time: {}sec".format(time.time() - start))
  os.remove("C:\\dropFile\\test\\A\\01Intro.pdf")
  print("update with a deleted file...")
  start = time.time()
  # D,V,S,T = prepare_env(args.root_path, T, verbose=False)
  print("elapsed time: {}sec".format(time.time() - start))
  recommendation_path = dropfile(args.input_file, args.root_path, None, None, None, verbose=True)
  print("elapsed time: {}sec".format(time.time()-start))
  print("Execution Result: {}".format(recommendation_path))