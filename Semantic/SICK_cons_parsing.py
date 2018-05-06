from nltk.parse.stanford import StanfordParser
from nltk.internals import find_jars_within_path
from nltk import tree

split=["trial","train","test_annotated"]
for s in split:
    f=open("SICK_"+s+".txt","r")
    lines=f.readlines()
    sentences=[]
    labels=[]
    for i in range(1,len(lines)):
        a=lines[i].split("\t")
        sentences.extend([a[1],a[2]])
        labels.extend([a[3],a[3]])



    parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    stanford_dir = parser._classpath[0].rpartition('/')[0]
    parser._classpath = tuple(find_jars_within_path(stanford_dir))
    parser.java_options = '-mx5000m' # To increase the amount of RAM it can use.
    #a=[parse.tree()._pformat_flat("","()",False) for parse in parser.raw_parse("The young boys are playing outdoors and the man is smiling nearby")]
    a = [[parse for parse in dep_graphs] for dep_graphs in parser.raw_parse_sents(sentences)]
    file=open("SICK_cons_parse_"+s+".txt","w")
    for i in range(len(a)):
        for j in range(len(a[i])):
            a[i][j].chomsky_normal_form(horzMarkov=1)
            a[i][j].collapse_unary(collapsePOS=True)
            d=a[i][j]._pformat_flat("","()",False)
            sent1=d.replace("ROOT",labels[i],1)
            file.write(sent1 + "\n")
    file.close()
    f.close()
