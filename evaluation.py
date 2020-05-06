# evaluation code
import argparse
import os
import random
import shutil
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
def prepare_env(file_list: list, locate_flag: list):
  current_path = os.getcwd()
  test_path = current_path + "\\eval\\"

  label = ["" for _ in range(len(file_list))]

  find_common_parent_dir = []
  if test_path not in os.listdir(current_path):
    os.mkdir(test_path)

  try:
    # 가장 상위의 공통 디렉토리를 찾는다.
    for file_name in file_list:
      find_common_parent_dir.append(os.path.split(file_name)[0].split("\\"))  # \n이나 /를 기준으로 자른다

    compare_dir = find_common_parent_dir[0]
    for temp_dir in find_common_parent_dir[1:]:
      print(compare_dir)
      if len(compare_dir) < len(temp_dir):
        continue
      elif len(compare_dir) == len(temp_dir):
        if compare_dir[-1] != temp_dir[-1]:
          compare_dir = compare_dir[:-1]
        else:
          continue
      else:
        compare_dir = temp_dir

    # 가장 상위의 공통 디렉토리
    common_parent_dir = "\\".join(compare_dir)

    for idx, file_name in enumerate(file_list):
      file_dir = os.path.split(file_name)[0]
      # 파일이 common_parent_dir보다 하위 디렉토리에 있다면, 하위 디렉토리들을 생성한 후 copy
      if file_dir != common_parent_dir:
        a = file_dir.split("\\")
        b = common_parent_dir.split("\\")
        additional_path = a[len(b):]
        temp_path = test_path[:-1]
        for i in range(len(additional_path)):
          subdir = temp_path + "\\" + additional_path[i]
          if additional_path[i] not in os.listdir(temp_path):
            os.mkdir(subdir)
          temp_path = subdir

        label[idx] = subdir

        if locate_flag[idx]:
          shutil.copy2(file_name, subdir)
      else:
        shutil.copy2(file_name, test_path)
        label[idx] = test_path

  except PermissionError:
    pass

  testset = file_list
  return test_path, label, testset


# function : evaluation이 이루어질 모든 경우의 location_flag 리스트를 구한다
# input : file_list
# output : list of locate_flag
# implementation : output의 각 element는 위 prepare_env 함수의 locate_flag로 들어갈 수 있는 포맷이어야함
def calculate_combination(file_list):
  global INITIAL_TEST_FRAC
  temp_dict = {}
  locate_flag = [False for _ in range(len(file_list))]
  for file_name in file_list:
    folder_name = os.path.split(file_name)[0]
    if folder_name in temp_dict:
      temp_dict[folder_name].append(file_name)
    else:
      temp_dict[folder_name] = [file_name]

  for folder_name in temp_dict:
    chosen_files = random.sample(temp_dict[folder_name], round(len(temp_dict[folder_name]) * INITIAL_TEST_FRAC))
    for file in chosen_files:
      locate_index = file_list.index(file)
      locate_flag[locate_index] = True

  return locate_flag


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