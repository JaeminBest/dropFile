from pdfminer.high_level import extract_text
from preprocessing.preprocessing import Preprocessing
import pickle
from tqdm import tqdm

preprocessing = Preprocessing()
root_path = './test'
directory_dict={}
dir_hierarchy = preprocessing.lookup_directory(root_path, directory_dict)
file_list = list()
dir_list = list()
label_num = 0
for tar_dir in dir_hierarchy:
    file_list += dir_hierarchy[tar_dir]

for file_path in tqdm(file_list):
    text = extract_text(file_path)
    new_path = file_path[:-4]
    with open(new_path, 'wb') as f:
        pickle.dump(text, f)

    