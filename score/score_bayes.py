import argparse
import time
import numpy as np
from collections import defaultdict
import os

def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y

# ===============================================================================================
# NaiveBayes class 
# I asssume that priors are same for all directories. So I only considered likelihood of words. 
# ===============================================================================================

class NaiveBayes():
    def __init__(self, num_vocab, num_classes):
        if 'DROPFILE_LOGLEVEL' in os.environ:
            self.verbose = int(os.environ['DROPFILE_LOGLEVEL'])
        else:
            self.verbose = 0

        self.num_vocab = num_vocab
        self.num_classes = num_classes

        self.class_and_word_to_counts = np.zeros((self.num_classes, self.num_vocab))

        self.log_likelihood = None

    # Get likelihood, P(w|c) with smoothing
    # input - bows : ndarray of shape (num_classes, num_vocab) 
    # output - log_likelihood : ndarray of  P with shape (num_classes, num_vocab) 
    # where P[c,w] is the log likelihood of word w and given class c. 
    def get_likelihood_with_smoothing(self, bows):
        
        self.class_and_word_to_counts = np.asarray(bows)

        likelihood_array = np.zeros((self.num_classes, self.num_vocab))
        for i in range(len(likelihood_array)):
            likelihood_array[i] = (self.class_and_word_to_counts[i] + 1) / np.sum(self.class_and_word_to_counts[i]+1)
            

        self.log_likelihood = np.log(likelihood_array)

    # predict class (directory) by posterior,  p(c_k|w_1, ..., w_n) ~ sum(log(p(w_i|c_k)))
    # input - bow : bow you want to infer the class 
    # output - index : index of class that has highest posterior.
    def predict(self, bow):

        prob = [0]* self.num_classes

        for i in range(len(prob)):
            prob[i] = np.sum(bow * self.log_likelihood[i])

            if self.verbose:
                print("posterior {}-th : {} ".format(i,prob[i]))

        if prob != []:
            index = prob.index(max(prob))
        else:
            index = 0

        return index, prob


# main body of program: DropFile
# input : input file path, root path 
# output : recommended path
def score_bayes(input_file, root_path: str, preprocessing, DTM=None, vocab=None, synonym_dict=None, mse=False):
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
            doc_list.append(preprocessing.file2tok(file))
        vocab, synonym_dict = preprocessing.build_vocab(doc_list)
        # preprocessing : build DTM of files under root_path
        DTM = preprocessing.build_DTM(doc_list, vocab, synonym_dict)
    
    # accumulate DTM by label (directories)
    label_DTM = list()
    offset = 0
    for label, tar_dir in enumerate(dir_list):
        file_num = len(dir_hierarchy[tar_dir])
        temp_dtm = np.zeros(len(vocab))

        for j in range(file_num):
            temp_dtm += np.asarray(DTM[offset+j])
            
        label_DTM.append(temp_dtm)
        offset += file_num

    # preprocessing : build BoW, DTM score of input file
    dtm_vec = preprocessing.build_DTMvec(input_file, vocab, synonym_dict)

    # make NaiveBayes instance
    naivebayes = NaiveBayes(len(vocab.keys()), len(dir_list))

    # compute likelihood of NaiveBayes
    naivebayes.get_likelihood_with_smoothing(label_DTM)

    # predict the directory 
    index, prob = naivebayes.predict(dtm_vec)
    if index < len(dir_list):
        dir_path = dir_list[index]

    return dir_list, prob, DTM, vocab, synonym_dict


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
    recommendation_path = dropfile(args.input_file, args.root_path)
    print("elapsed time: {}sec".format(time.time()-start))
    print("Execution Result: {}".format(recommendation_path))