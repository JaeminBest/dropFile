# DropFile (CS372 NLP Term Project)


### Quick Start

```bash
$ mkdir test (그 후 여기에 테스트해볼 파일들을 저장해야함)
$ pip install -r requirement.txt
$ python -m spacy download en_core_web_lg
$ python3 dropfile.py -r (root_path: default ./test) -i file_path
$ python3 evaluation.py -r (root_path: defulat ./test) -f (full evaluation)
```
*./test* 디렉토리에 잘 분류된 파일들을 위치시켜야한다. 용량 상의 문제로 해당 레포에는 tests 파일은 올리지 않는다.  
&nbsp;


### 프로그램 구조
- dropfile.py : 단일 파일 디렉토리 경로 추천
    - function dropfile() : 경로 추천
    - function prepare_env() : 사전 계산 후, DTM, vocab, synonym_dict 반환
- evaluation.py : 정확도 측정을 위한 평가 코드
- preprocessing
    - preprocessing.py : 각 파일의 특성을 추출하는 전처리 코드
        - class Preprocessing : 원래 버전
        - class DependencyParserPreprocessing : head of dependency structure만 가져오는 버전
        - class NounPhrasePreprocessing : head of noun phrase만 가져오는 버전
        - class NounPreprocessing : 건호님 버전
        - class SpacyPreprocessing : 윤석님 버전
- score
    - score_cosine.py : cosine similarity로 예측 (원래 버전)
    - score_mse.py : mse로 예측
    - score_bayes.py : naive bayes로 예측
- tests : 테스트용 파일 (root_path로 설정되어 있음)  
&nbsp;  
   
### 중간 역할 분배 (20.05.02)
(0) 스켈레톤 코드 작성
(1) preprocessing.py - file2text, text2tok 담당 : pdfminer 라이브러리, nltk 라이브러리 능숙자  
(2) preprocessing.py - tok2bow, build_DTM, build_DTMvec 담당 : nltk 라이브러리, 알고리즘 부분 능숙자  
(3) preprocessing.py - lookup_directory, evaluation.py - prepare_env, calculate_combination 담당 : os 라이브러리 능숙자  
(4) pipelining, debugging 담당 : python 프로그래밍에 능숙한 사람, 짧지만 주기적으로 관리할 수 있는 사람

- 김재민 : (0) + (4)
- 이윤석 : 
- 김건호 : 
- 박범식 : 
- 박창현 : 