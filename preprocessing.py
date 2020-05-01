# preprocessing code
import os
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO


# function : lookup directory hierarchy under root_path
# input : root_path
# output : dictionary, 
#          key = 분류할 디렉토리의 절대경로, 
#          value = 그(key) 디렉토리 안에 있는 file_name(절대경로)의 리스트 
# output : dictionary that having absolute directory path as key 
#          and list of file name inside the directory as value
# implementation : os 라이브러리 사용하면 될듯, UNIX 환경 가정 (mac, ubuntu)
def lookup_directory(root_path: str):
  pass


# function : read pdf file and convert into text(string) format
# input : file_path
# output : text
# implementation : pdfminer 라이브러리 사용 (ref: https://lsjsj92.tistory.com/304)
def fiile2text(file_path: str):
  pass


# function : convert text into list of parsed token
# input : text (불용어, 문장부호, 띄어쓰기 등 포함)
# output : list of parsed token, 의미를 가지는 토큰만 포함한 리스트
# implementation : Regex 라이브러리로 필터링
def text2tok(text: str):
  pass


# function : convert token list into BoW
# input : list of token
#         ex) ["I", "love", "her", ...]
# output : BoW
#         ex) {"I": 2, "love": 1, "her": 1, ....}
def tok2bow(token_list: str):
  pass
  

# function : build BoW
def build_BoW(file_path: str):
  txt = file2txt(file_path)
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
  vocab_list = None
  DTM = None
  return vocab_list, DTM


# function : build vocab list out of each list_of parsed token
# input : single BoW of input_file and vacab list from files under root path
# output : list(not ndarray) of total vocab list, DTM
def build_DTMvec(bow, vocab_list):
  return 
