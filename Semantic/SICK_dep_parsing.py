from nltk.parse.stanford import StanfordNeuralDependencyParser

split=["train","trial","test_annotated"]
for s in split:
    f=open("SICK_"+s+".txt","r")
    lines=f.readlines()
    sentences=[]
    labels=[]
    for i in range(1,len(lines)):
        a=lines[i].split("\t")
        sentences.extend([a[1],a[2]])
        labels.extend([a[3],a[3]])



    parser = StanfordNeuralDependencyParser(model_path="edu/stanford/nlp/models/parser/nndep/english_UD.gz")
    stanford_dir = parser._classpath[0].rpartition('/')[0]
    slf4j_jar = stanford_dir + '/slf4j-api.jar'
    parser._classpath = list(parser._classpath) + [slf4j_jar]
    parser.java_options = '-mx5000m' # To increase the amount of RAM it can use.
    file=open("SICK_dep_parse_"+s+".txt","w")
    #a=[parse.tree()._pformat_flat(" ","()",False) for parse in parser.raw_parse("The young boys are playing outdoors and the man is smiling nearby")]

    parsed_sents = [[parse.tree()._pformat_flat("","()",False) for parse in dep_graphs] for dep_graphs in parser.raw_parse_sents(sentences)]
    for i in range(len(parsed_sents)):
        for j in range(len(parsed_sents[i])):
            sent1="("+labels[i]+" "+parsed_sents[i][j]+")"
            file.write(sent1 + "\n")
    file.close()
    f.close()
