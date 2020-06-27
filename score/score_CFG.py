# Context-Free grammar dropfile implementation


import argparse
import time
import numpy as np
from collections import defaultdict
import os

# cosine similarity
def cosine_similarity(A,B):
  ndA = np.asarray(A)
  ndB = np.asarray(B)
  a = np.dot(ndA,ndB)
  b = (np.linalg.norm(ndA)*np.linalg.norm(ndB))
  if b == 0:
    return 0
  return np.dot(ndA,ndB)/(np.linalg.norm(ndA)*np.linalg.norm(ndB))

def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  return y



# main body of program: DropFile
# input : input file path, root path 
# output : recommended path
def score_CFG(input_file: str, root_path: str, preprocessing, DTM=None, vocab=None, synonym_dict=None, mse=False):
  if 'DROPFILE_LOGLEVEL' in os.environ:
    verbose = int(os.environ['DROPFILE_LOGLEVEL'])
  else:
    verbose = 0

  # preprocessing : lookup hierarchy of root path
  directory_dict = defaultdict(list) # empty dictionary for lookup_directory function
  dir_hierarchy = preprocessing.lookup_directory(root_path, directory_dict) # change it to have 2 parameter

  file_list = list()
  dir_list = list() 
  label_num = 0
  for tar_dir in dir_hierarchy:
    file_list += dir_hierarchy[tar_dir]
    dir_list.append(tar_dir)
    label_num += 1
    
  # preprocessing : build vocabulary from file_list
  if (DTM is None) and (vocab is None) and (synonym_dict is None):
    doc_list = list()
    for file in file_list:
      doc_list.append(preprocessing.extract_mean(file))
    vocab, synonym_dict = preprocessing.build_vocab(doc_list)
    
    # preprocessing : build DTM of files under root_path
    DTM = preprocessing.build_DTM(doc_list, vocab, synonym_dict)
    
  # preprocessing : build BoW, DTM score of input file
  dtm_vec = preprocessing.build_DTMvec(input_file, vocab, synonym_dict)
  
  # similarity calculation using cosine similarity
  sim_vec = list()
  for i in range(len(DTM)):
    sim_vec.append(cosine_similarity(DTM[i],dtm_vec)) # evaluate similairty by cosin_similarity
    
  # calculate label score by getting maximum value of the directory
  # result will be score of each directory
  label_score = [0.0 for i in range(label_num)]
  offset = 0
  for label, tar_dir in enumerate(dir_list):
    file_num = len(dir_hierarchy[tar_dir])
    temp_max = 0.0
    for j in range(file_num):
      temp_score = sim_vec[offset+j]
      if temp_score > temp_max:
        temp_max = temp_score
      
    label_score[label] = temp_max
    offset += file_num
  
  # calculate the softmax score
  soft_score = softmax(label_score)

  # print each directory's softmax score
  for i in range(len(dir_list)):
    if verbose:
      print("directory %s score : %6.3f" %(dir_list[i],soft_score[i]))

  return dir_list, soft_score, DTM, vocab, synonym_dict
  


# main execution command
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='dropFile program')
  parser.add_argument('-r', '--root-path', help='root path that input file should be classified into',
                      type=str, action='store', default='./test')
  parser.add_argument('-i', '--input-file', help='input file initial path',
                      type=str, action='store')
  args = parser.parse_args()
  print('root path : {}, input file: {}'.format(args.root_path, args.input_file))
  if (args.input_file is None):
    parser.error("--input-file(-i) format should be specified")
  
  print("Running DropFile...")
  start = time.time()
  softmax= dropfile(args.input_file, args.root_path)
  print("elapsed time: {}sec".format(time.time()-start))
  # print("Execution Result: {}".format(recommendation_path))