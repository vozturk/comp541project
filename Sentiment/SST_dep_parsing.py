# -*- coding: utf-8 -*-
from nltk.parse.stanford import StanfordNeuralDependencyParser
import sys
reload(sys)
sys.setdefaultencoding('UTF-8')

split=["train","dev","test"]
for s in split:
    f=open(s+".txt","r")
    lines=f.readlines()
    if s=="train":
        del lines[2874]
        del lines[4340]
        del lines[5796]
        del lines[7405]
        del lines[8006]
    sentences=[]
    for i in range(len(lines)):
        a1=lines[i].replace("\n","")
        a2=a1.replace("(","")
        a3=a2.replace("0 ","")
        a4=a3.replace("1 ","")
        a5=a4.replace("2 ","")
        a6=a5.replace("3 ","")
        a7=a6.replace("4 ","")
        a8=a7.replace(")","")
        #a9=a8.replace("\/","/")

        sentences.append(a8)
    #print(sentences[2874])

    parser = StanfordNeuralDependencyParser(model_path="edu/stanford/nlp/models/parser/nndep/english_UD.gz")
    stanford_dir = parser._classpath[0].rpartition('/')[0]
    slf4j_jar = stanford_dir + '/slf4j-api.jar'
    parser._classpath = list(parser._classpath) + [slf4j_jar]
    parser.java_options = '-mx5000m' # To increase the amount of RAM it can use.
    file=open("SST_dep_parse_"+s+".txt","w")
    parsed_sents=parser.raw_parse_sents(sentences)
    parsed_sents = [[parse.tree()._pformat_flat("","()",False) for parse in dep_graphs] for dep_graphs in parser.raw_parse_sents(sentences)]

    for sent in parsed_sents:
        file.write(sent[0] + "\n")
    file.close()
    print ("data"+s)
    f.close()
