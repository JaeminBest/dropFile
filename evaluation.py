# evaluation code
import argparse
import os
import preprocessing
import dropfile
from tqdm import tqdm

INITIAL_TEST_FRAC = 0.6
INITIAL_PATH = './test'

# function : prepare environment, build new root_path and relocate each file
# input : file_list, locate_flag
#         ex) file_list : ["/test/nlp-1.pdf","/test/nlp-2.pdf",...]
#             locate_flag : [true,false,true,true, ...] 
#                (= 해당 인덱스의 파일이 initial로 존재할지, 혹은 test용
#                  input으로 사용될지에 대한 flag)
# output : test_path (test가 진행될 root_path, 새로운 디렉토리에 re-locate시켜주어야함.
#                     test 디렉토리에서 locate_flag가 true인 것들에 대해서만 새로운 root_path(e.g. /eval 디렉토리)로 복사해주어야함)
#          label (test의 input으로 들어갈 파일들에 대한 올바른 결과값에 대한 리스트)
#          testset (test에 사용될 파일들의 절대경로 리스트, e.g. ["/test/nlp-1.pdf",..], 원래 파일들이 있었던 경로로 지정)
# implementation : os 라이브러리 사용
def prepare_env(file_list, locate_flag):
  testset = list()
  label = list()
  test_path = None
  return test_path, label, testset


# function : evaluation이 이루어질 모든 경우의 location_flag 리스트를 구한다
# input : file_list
# output : list of locate_flag
# implementation : output의 각 element는 위 prepare_env 함수의 locate_flag로 들어갈 수 있는 포맷이어야함
def calculate_combination(file_list):
  return 


# evaluate for experiment
def evaluation(root_path: str):
  # preprocessing : lookup hierarchy of root path
  dir_hierarchy = preprocessing.lookup_directory(root_path)
  file_list = list()
  dir_list = list()
  label_num = 0
  for tar_dir in dir_hierarchy:
    file_list += dir_hierarchy[tar_dir]
    dir_list.append(tar_dir)
    label_num += 1
  
  # calculate combination
  combination = calculate_combination(file_list)
  
  # start evaluation
  print("Start evaluation..")
  trial = 0
  correct = 0
  for locate_flag in tqdm(combination,mininterval=1):
    test_path, label, testset = prepare_env(file_list,locate_flag)
    for i,input_path in enumerate(testset):
      trial +=1
      answer = dropfile.dropfile(input_path,test_path)
      if (answer==label[i]):
        correct += 1
  print("Evaluation Result: {}/{}".format(correct,trial))


# main execution command
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='dropFile evaluation program')
  parser.add_argument('-r', '--root-path', help='root path that input file should be classified into',
                      type=str, action='store', default='./test')
  parser.add_argument('-f', help='full evaluation, compute for all combination',
                      type=str, action='store_true')
  args = parser.parse_args()
  
  print("Running Evaluation DropFile...")
  start = time.time()
  evaluation(args.root_path)
  print("elapsed time: {}sec".format(time.time()-start))
  print("Execution Result: {}".format(recommendation_path))