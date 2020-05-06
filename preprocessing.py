# preprocessing code
import os
import re
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.high_level import extract_text
from io import StringIO
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()





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
      return
  except PermissionError:
    pass
  return


# function : read pdf file and convert into text(string) format
# input : file_path
# output : text
# implementation : pdfminer 라이브러리 사용 (ref: https://lsjsj92.tistory.com/304)
def file2text(file_path: str):
  text = extract_text(file_path) # extract text from pdf file
  return text


# function : convert text into list of parsed token
# input : text (불용어, 문장부호, 띄어쓰기 등 포함)
# output : list of parsed token, 의미를 가지는 토큰만 포함한 리스트
# implementation : Regex 라이브러리로 필터링
def text2tok(text: str):
  words = word_tokenize(text) # tokenize words by nltk word_toknizer
  stops = stopwords.words('english')
  words = [word.lower() for word in words] # convert uppercase to lowercase
  words = [word for word in words if word not in stops] # remove stopwords
  # words = [word for word in words if word.isalnum() and (not word.isnumeric())] # filter non-alphanumeric words
  words = [word for word in words if re.match('^[a-zA-Z]\w+$', word)] # regex version of above line
  words = [lm.lemmatize(word) for word in words] # lemmatize words
  return words


# function : convert token list into BoW
# input : list of token
#         ex) ["I", "love", "her", ...]
# output : BoW
#         ex) {"I": 2, "love": 1, "her": 1, ....}
def tok2bow(token_list: str):
  return {"sample": 1, "text": 1}
  

# function : build BoW
def build_BoW(file_path: str):
  txt = file2text(file_path)
  tok = text2tok(txt)
  bow = tok2bow(tok)
  return bow


# function : build vocab list out of each list_of parsed token
# input : list of BoW
#        ex) [{"I": 2, "love": 1, "her": 1, ....}(1st file),
#             {"I": 2, "love": 1, "her": 1, ....}(2nd file),{..},{..}]
# output : list of total vocab list, DTM
#         ex) vocab_list : ["I", "love", "her"]
#         ex) DTM : [[1,2,4,,...](1st file),[3,5,2,...](2nd file), ...]
def build_DTM(bow_list):
  vocab_list = ["sample", "text"]
  DTM = [[1,2],[2,0]]
  return vocab_list, DTM


# function : build vocab list out of each list_of parsed token
# input : single BoW of input_file and vacab list from files under root path
# output : list(not ndarray) of total vocab list, DTM
def build_DTMvec(bow, vocab_list):
  return [1,0]

if __name__ == "__main__":
  test_file_path = "C://dropFile/test/nlp.pdf" # change it for your own test file
  text = file2text(test_file_path)
  words = text2tok(text)
  print(words)