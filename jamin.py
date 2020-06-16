import csv
import spacy
import re
import argparse
import nltk
from spacy.lang.en import English
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import word_tokenize
import time

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

depParser = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

OUTPUT_PATH = './CS372_HW4_output_20170148.csv'
TAG_LIST = [".",",","-LRB-","-RRB-","``","\"\"","''",",","$","#","AFX","CC","CD","DT","EX","FW","HYPH","IN","JJ","JJR","JJS","LS","MD","NIL","NN","NNP","NNPS","NNS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB","ADD","NFP","GW","XX","BES","HVS","_SP"]
IGNORE_LIST = [".",",","$","-LRB-","-RRB-","``","\"\"","\'\'","PRP$","WP$"]
CONVERT_LIST = ["PERIOD","COMMA","UNK","UNK","UNK","QUOTE","QUOTE","QUOTE","UNK","UNK"]
# CONVERT_LIST = ["PERIOD","COMMA","DOLLAR","LRB","RRB","BQUOTE","QUOTE","DQUOTE","PRPS","WPS"]
KEYWORD_LIST = ['activate','inhibit','bind','accelerate','augment','induce','stimulate','require','up-regulate','abolish','block','down-regulate','prevent']

default_grammar = r"""
S -> PHRASE2 "PERIOD"
PHRASE1 -> VP | NP VP | VP VP | "IN" PHRASE2
PHRASE2 -> PHRASE1 "IN" PHRASE2 | PHRASE1 "THAT" PHRASE2 | PHRASE1 "COMMA" PHRASE2 | PHRASE1
NP -> NP2 | NP2 "AND" NP | NP2 "OR" NP | NP2 "COMMA" NP2 "AND" NP2 | NP2 "COMMA" NP2 "COMMA" "AND" NP2
NP1 -> "NOUN" | "NOUN" NP1 | "VBN" NP1 | "VBG" NP1 | "QUOTE" PHRASE2 "QUOTE" | "QUOTE" NP "QUOTE"
NP2 -> NP1 | NP1 PP
VP -> VP1 | VP1 "AND" VP | VP1 "OR" VP | VP1 "COMMA" VP1
VP1 -> VERB PP | VERB NP | VERB | VP2 PP | VP2
VP2 -> "VERB" "VERB" | "VERB" "VBN" | "VERB" "VBG"
PP -> "IN" NP | "TO" VP1
VERB -> "BE" "VBN" | "BE" | "VERB" | "VBN" | "BE " "VBG"
"""

# traverse through nltk.Tree and change into string format
def formatting_traverse(t):
    try:
        label = t.label()
    except AttributeError:
        word = str(t).split('/')[0]
        temp = str(t).split('/')[1]
        temp = convert_single_tok(word,None,temp)
        if temp =='UNK':
            return ""
        return "{} ".format(temp)
    else:
        string = "({} ".format(label)
        for child in t:
            string+=formatting_traverse(child)
        string+=") "
        return string

# from train dataset, read annotated tagset and convert into prefered option of tag
# change each tag in preferarable way by traversing through whole tree node
def reduce_tag(tag):
    try:
        tree = nltk.Tree.fromstring(tag)
        tag = formatting_traverse(tree)
        return tag
    except:
        return None

# convert token into preferarable way
def convert_token(tokens):
    pos_sent = list()
    index_list = list()
    for i,tok in enumerate(tokens):
        temp = convert_single_tok(tok.text,tok.pos_,tok.tag_)
        if temp =='UNK':
            continue
        pos_sent.append(temp)
        index_list.append(i)
    return pos_sent,index_list

# convert single token into preferarable way
# only used tag is "VERB", "VBN", "NOUN", "IN", "COMMA", "THAT", "BE', ""UNK"
def convert_single_tok(word,pos,tag,level=2):
    rv = tag
    if level>=1:
        if word=='that':
            rv = 'THAT'
        elif tag in IGNORE_LIST and (level<3):
            idx = IGNORE_LIST.index(tag)
            rv = CONVERT_LIST[idx]
        elif (word in IGNORE_LIST) and (level<3):
            idx = IGNORE_LIST.index(word)
            rv = CONVERT_LIST[idx]
        elif pos=='VERB':
            if tag=='VBN':
                rv='VBN'
            elif tag=='VBG':
                rv = 'VBG'
            else:
                rv='VERB'
        elif lemmatizer.lemmatize(word,"v")=="be":
            rv = "BE"
        elif pos=='PRON' or tag.startswith("PR"):
            rv = 'NOUN'
        elif tag.startswith("NN"):
            rv = 'NOUN'
        elif tag.startswith("VB"):
            if tag=='VBN':
                rv='VBN'
            elif tag=='VBG':
                rv='VBG'
            else:
                rv='VERB'
        elif tag=="CC":
            if word.lower()=='and':
                rv = "AND"
            elif word.lower()=='or':
                rv = "OR"
            else:
                rv = 'UNK'
        elif tag=='TO':
            rv = 'TO'
        elif tag.startswith("IN"):
            rv="IN"
        else:
            rv = 'UNK'
    return rv

# function that parse <X,ACTION,Y> form
def gt_parser(string):
    string = string.replace("<","")
    string = string.replace(">","")
    string_list = string.split(',')
    new_list = list()
    for i in range(len(string_list)//3):
        new_list.append(tuple(string_list[i:i+3]))
    return new_list
    
def extract_np_outof_pp(subtree,tok_list):
    target_obj = None
    for leave in subtree:
        if type(leave) == nltk.tree.ParentedTree:
            if leave.label()=='NP':
                target_obj = extract_np(leave,tok_list)
                if target_obj is not None:
                    break
    return target_obj

def extract_np(subtrees,tok_list):
    target_obj = None
    for subtree in subtrees:
        if type(subtree) == nltk.tree.ParentedTree:
            if subtree.label().startswith('NP'):
                if (is_pure_ne_chunk(subtree)):
                    flat = str(subtree.flatten())
                    flat = flat.split()[1:]
                    flat = [tok_list[int(el.replace(')',"").split('/')[1])].text for el in flat]
                    flat = " ".join(flat)
                    return flat
                else:
                    target_obj = extract_np(subtree,tok_list)
                    if target_obj is not None:
                        return target_obj
                    else:
                        continue
            else:
                continue
    return target_obj


"""
helper function that identify corresponding subtree is only containing NOUN
which is NE chunk
"""
def is_pure_ne_chunk(subtrees):
    flag = True
    for subtree in subtrees:
        if type(subtree) == nltk.tree.ParentedTree:
            if subtree.label()=='NP1':
                flag = is_pure_ne_chunk(subtree)
                if not flag:
                    return flag
            elif subtree.label()=='NOUN':
                continue
            else:
                return False
        elif str(subtree).split('/')[0] !='NOUN':
            return False
        else:
            continue
    return flag
    
def find_keyword(subtree, tok_list):
    target_verb = None
    target_obj = None
    REVERSE_FLAG = False
    for leave in subtree:
        if leave.label()=='VERB':
            for terminal in leave:
                pos1 = str(terminal).split('/')[0]
                tok = tok_list[int(str(terminal).split('/')[1])]
                if pos1=='VERB' and (tok.lemma_ in KEYWORD_LIST):
                    target_verb = tok.text
                    break
            if target_verb is not None:
                for terminal in leave:
                    pos2 = str(terminal).split('/')[0]
                    if pos2=='BE':
                        REVERSE_FLAG = True
                        break
        if target_verb is not None:
            break
    
    if target_verb is not None:
        for leave in subtree:
            if leave.label()=='PP':
                target_obj = extract_np_outof_pp(leave,tok_list)
            elif leave.label()=='NP':
                target_obj = extract_np(leave,tok_list)
            else:
                continue
    
    return target_verb, target_obj, REVERSE_FLAG
    

"""
"""
def nltkTree_traverse_index(t,idx):
    try:
        label = t.label()
    except AttributeError:
        return "{}/{} ".format(t,idx),idx+1
    else:
        string = "({} ".format(label)
        for child in t:
            nstring,new_idx=nltkTree_traverse_index(child,idx)
            string += nstring
            idx = new_idx
        string+=") "
        return string, idx


"""
"""
def nltkTree_VP_search(subtrees,tok_list):
    target_verb = None
    target_obj = None
    target_subj = None
    REVERSE_FLAG = False
    temp = subtrees
    for subtree in subtrees:
        if type(subtree) == nltk.tree.ParentedTree:
            if subtree.label()=='VP1':
                target_verb,target_obj,REVERSE_FLAG = find_keyword(subtree,tok_list)
                if (target_verb is not None) and (target_obj is not None):
                    break
            elif subtree.label()=='VP':
                target_verb,target_obj,_,REVERSE_FLAG = nltkTree_VP_search(subtree,tok_list)
                if (target_verb is not None) and (target_obj is not None):
                    # this mean this SUBTREES is root phrase which contain subject word
                    parent = subtree.parent()
                    for sbt in parent.subtrees():
                        if type(sbt) == nltk.tree.ParentedTree:
                            if sbt.label()=='NP':
                                target_subj = extract_np(sbt,tok_list)
                                break
                    break
    return target_verb,target_obj,target_subj,REVERSE_FLAG

# main function that read train dataset and test dataset to solve probelm
def main(train_path:str,test_path:str,file_flag:bool,evaluate:bool):
    # trainset/testset preparation
    trainset = list()
    train_gt = list()
    with open(train_path,encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            trainset.append(row)
            train_gt.append(gt_parser(row[-1]))
    testset = list()
    test_gt = list()
    with open(test_path,encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            testset.append(row)
            test_gt.append(gt_parser(row[-1]))
    
    # build grammar automatically
    # By using annotation of pre-defined train dataset,
    #  I believe that there might be useful context-free grammar expression for whole sentence parser
    tag_dataset = list()
    for data in trainset:
        temp = reduce_tag(data[7])
        if temp is not None:
            tag_dataset.append(temp)
            
    ruleset = set(rule for tag_str in tag_dataset[:3]
                        for rule in nltk.Tree.fromstring(tag_str).productions())
    
    rulestring = "\n".join([str(rule) for rule in ruleset])
    grammar_d = nltk.CFG.fromstring(default_grammar)
    rd_parser_d = nltk.RecursiveDescentParser(grammar_d) 
    
    # iterate through train dataset and opitmize some option to evaluate better solution for testset
    if not evaluate:
        start = time.time()
        confusion_mat_train = [0,0,0,0] # TP, TN, FP, FN
        for i,data in enumerate(trainset):
            new_sent = data[6]
            tokens = depParser(new_sent)
            pos_sent,index_list = convert_token(tokens)
            tok_list = [tokens[idx] for idx in index_list]
            
            tree_list = list()
            # print("=========================== {}sec".format(float(time.time())-float(start)))
            start = time.time()
            for tree in rd_parser_d.parse(pos_sent):
                tree_list.append(tree)
            # print(new_sent)
            # print("scenario 0: parsed tree = {}".format(len(tree_list)))
            answer = list()
            
            # scenario 0 : there is matching grammar
            target_verb = None
            target_obj = None
            target_subj = None
            REVERSE_FLAG = False
            temp = list()
            for tree in tree_list:
                nstring,_ = nltkTree_traverse_index(tree,0)
                ntree = nltk.Tree.fromstring(nstring)
                ntree = nltk.tree.ParentedTree.convert(ntree)
                target_verb,target_obj,target_subj,REVERSE_FLAG = nltkTree_VP_search(ntree.subtrees(),tok_list)
                if (target_obj is not None) and (target_subj is not None) and (target_verb is not None):
                    temp.append((target_subj,target_verb,target_obj))
            temp = list(set(temp))
            if len(temp)>0:
                answer = temp[0]
            else:
                # scenario 1 : dependency arrow go out from target keyword
                for idx,token in enumerate(tokens):
                    candidate = list()
                    REVERSE_FLAG = False # reverse flag
                    if (token.lemma_ in KEYWORD_LIST) and (token.pos_=="VERB"):
                        if token.tag_=='VBN':
                            for j in range(idx):
                                target = tokens[j]
                                if (target.lemma=='be') and (target.head==token):
                                    REVERSE_FLAG = True
                                    break
                        temp = [list(),"",list()]
                        temp[1] = token.text
                        curpos = token.i
                        for child in token.children:
                            if child.pos_=="NOUN":
                                if child.i<token.i:
                                    idx1 = 2 if REVERSE_FLAG else 0
                                    temp[idx1].append(extract_near_entity(child.i,tokens)) # string form
                                elif child.i> token.i:
                                    idx2 = 0 if REVERSE_FLAG else 2
                                    temp[idx2].append(extract_near_entity(child.i,tokens)) # string form
                        
                        candidate = reduce_XY(temp,tokens)
                        if candidate is not None:
                            answer = candidate
            
            try:
                GT = train_gt[i][0]
            except:
                GT = None
            print("ground truth:",GT)
            print("prediction:",answer)
            if (len(answer)!=0) and (answer is not None):
                if is_similar(answer,GT): #TP
                    print("TP")
                    confusion_mat_train[0]+=1
                else: #TN
                    print("TN")
                    confusion_mat_train[1]+=1
            else:
                if GT is None: #FP
                    print("FP")
                    confusion_mat_train[2]+=1
                else: #FN
                    print("FN")
                    confusion_mat_train[3]+=1
        
        train_TP = confusion_mat_train[0]
        train_TN = confusion_mat_train[1]
        train_FP = confusion_mat_train[2]
        train_FN = confusion_mat_train[3]
        
        train_prec = train_TP/(train_TP+train_FP)
        train_recall = train_TP/(train_TP+train_FN)
        train_F = 2*train_prec*train_recall / (train_prec+train_recall)
        print("========= TRAINSET RESULT OUTPUT ==========")
        print(train_prec,train_recall,train_F)



    # # testset evaluation
    else:
        result = list()
        start = time.time()
        confusion_mat_test = [0,0,0,0] # TP, TN, FP, FN
        for i,data in enumerate(testset):
            new_sent = data[6]
            tokens = depParser(new_sent)
            pos_sent,index_list = convert_token(tokens)
            tok_list = [tokens[idx] for idx in index_list]
            
            tree_list = list()
            # print("=========================== {}sec".format(float(time.time())-float(start)))
            start = time.time()
            for tree in rd_parser_d.parse(pos_sent):
                tree_list.append(tree)
            # print(new_sent)
            # print("scenario 0: parsed tree = {}".format(len(tree_list)))
            answer = list()
            
            # scenario 0 : there is matching grammar
            target_verb = None
            target_obj = None
            target_subj = None
            REVERSE_FLAG = False
            temp = list()
            for tree in tree_list:
                nstring,_ = nltkTree_traverse_index(tree,0)
                ntree = nltk.Tree.fromstring(nstring)
                ntree = nltk.tree.ParentedTree.convert(ntree)
                target_verb,target_obj,target_subj,REVERSE_FLAG = nltkTree_VP_search(ntree.subtrees(),tok_list)
                if (target_obj is not None) and (target_subj is not None) and (target_verb is not None):
                    temp.append((target_subj,target_verb,target_obj))
            temp = list(set(temp))
            if len(temp)>0:
                answer = temp[0]
            else:
                # scenario 1 : dependency arrow go out from target keyword
                for idx,token in enumerate(tokens):
                    candidate = list()
                    REVERSE_FLAG = False # reverse flag
                    if (token.lemma_ in KEYWORD_LIST) and (token.pos_=="VERB"):
                        if token.tag_=='VBN':
                            for j in range(idx):
                                target = tokens[j]
                                if (target.lemma=='be') and (target.head==token):
                                    REVERSE_FLAG = True
                                    break
                        temp = [list(),"",list()]
                        temp[1] = token.text
                        curpos = token.i
                        for child in token.children:
                            if child.pos_=="NOUN":
                                if child.i<token.i:
                                    idx1 = 2 if REVERSE_FLAG else 0
                                    temp[idx1].append(extract_near_entity(child.i,tokens)) # string form
                                elif child.i> token.i:
                                    idx2 = 0 if REVERSE_FLAG else 2
                                    temp[idx2].append(extract_near_entity(child.i,tokens)) # string form
                        
                        candidate = reduce_XY(temp,tokens)
                        if candidate is not None:
                            answer = candidate
            
            try:
                GT = test_gt[i][0]
            except:
                GT = None
            print("ground truth:",GT)
            print("prediction:",answer)
            if (len(answer)!=0) and (answer is not None):
                if is_similar(answer,GT): #TP
                    print("TP")
                    confusion_mat_test[0]+=1
                else: #TN
                    print("TN")
                    confusion_mat_test[1]+=1
            else:
                if GT is None: #FP
                    print("FP")
                    confusion_mat_test[2]+=1
                else: #FN
                    print("FN")
                    confusion_mat_test[3]+=1
            result.append((data[0],data[1],data[2],data[3],data[4],
                           data[5],data[6],data[7],data[8],answer))
        
        test_TP = confusion_mat_test[0]
        test_TN = confusion_mat_test[1]
        test_FP = confusion_mat_test[2]
        test_FN = confusion_mat_test[3]
        
        try:
            test_prec = test_TP/(test_TP+test_FP)
        except:
            test_prec = 0
        try:
            test_recall = test_TP/(test_TP+test_FN)
        except:
            test_recall = 0
        test_F = 2*test_prec*test_recall / (test_prec+test_recall)
        print("========= TESTSET RESULT OUTPUT ==========")
        print(test_prec,test_recall,test_F)
    
        # print as output
        fw = open(OUTPUT_PATH, 'w', encoding='utf-8', newline='')
        wr = csv.writer(fw)
        for row in result:
            wr.writerow(row)
        fw.close()
        return

"""
function that compare ANSWER and GT and find out this is similar or not.
"""
def is_similar(answer,gt):
    a1,b1,c1 = answer
    a2,b2,c2 = gt
    
    FLAG_A = False
    a1t = a1.split()
    a2t = a2.split()
    for el2 in a2t:
        for el1 in a1t:
            if el2.find(el1)!=-1 or el1.find(el2)!=-1:
                FLAG_A = True
                break
    
    FLAG_B = False
    b1t = b1.split()
    b2t = b2.split()
    for el2 in b2t:
        for el1 in b1t:
            if el2.find(el1)!=-1 or el1.find(el2)!=-1:
                FLAG_B = True
                break
    
    FLAG_C = False
    c1t = c1.split()
    c2t = c2.split()
    for el2 in c2t:
        for el1 in c1t:
            if el2.find(el1)!=-1 or el1.find(el2)!=-1:
                FLAG_C = True
                break

    if FLAG_A and FLAG_B and FLAG_C:
        return True
    return False

"""
based on grammar knowledge Named entity = Noun + Noun + ..
"""
def extract_near_entity(idx,tokens):
    result = list()
    result.append(tokens[idx].text)
    LEFT_END = False
    nidx = idx
    while not LEFT_END:
        nidx-=1
        ntok = tokens[nidx]
        if ntok.pos_=='NOUN':
            result.insert(0,ntok.text)
        else:
            LEFT_END=True
    RIGHT_END = False
    nidx = idx
    while not RIGHT_END:
        nidx+=1
        ntok = tokens[nidx]
        if ntok.pos_=='NOUN':
            result.append(ntok.text)
        else:
            RIGHT_END = True
    return " ".join(result)

def reduce_XY(candidate,tokens):
    new_cand = list()
    try:
        new_cand[0] = candidate[0][0]
        new_cand[1] = candidate[1]
        new_cand[2] = candidate[2][0]
        return new_cand
    except:
        return None
    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',type=str,default='./train.csv')
    parser.add_argument('--test-file',type=str)
    parser.add_argument('--test-input',type=str)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    
    print("====== START program ======")
    if not args.test_input:
        FILE_FLAG = True
        main(args.train, args.test_file,FILE_FLAG,args.evaluate)
    elif args.test_file:
        FILE_FLAG = False
        main(args.train, args.test_input,FILE_FLAG,args.evaluate)
    else:
        print("ERROR: you should provide either '--test-file' or '--test-input' as argumennt")
    print("====== END program ======")