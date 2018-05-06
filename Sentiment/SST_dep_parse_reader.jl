for p in ("ZipFile","JLD","HDF5")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using ZipFile
using JLD

const TREEBANK_URL = "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"
const TREEBANK_SPLITS = ("train","dev", "test")
const nonglove=randn(Float32,(300,1))




mutable struct SentimentTree
    label
    children
    data
    tag

    SentimentTree(x) = new(x,nothing,nothing,nothing)
    SentimentTree(x,y) = new(x,y,nothing,nothing)
end

const SentimentTrees = Array{SentimentTree,1}


function isleaf(t::SentimentTree)
    return t.children == nothing
end

function pretty(t::SentimentTree)
    t.children == nothing && return t.label
    join([t.label; map(pretty, t.children)], " ")
end

let
    items = []

    global leaves
    function leaves(t::SentimentTree)
        empty!(items)
        helper(t)
        return items
    end

    function helper(subtree)
        if isleaf(subtree)
            push!(items, subtree)
        else
            for child in subtree.children
                helper(child)
            end
        end
    end
end

let
    nodes = []
    global nonterms
    function nonterms(t::SentimentTree)
        empty!(nodes)
        helper(t)
        return nodes
    end

    function helper(subtree)
        if !isleaf(subtree)
            push!(nodes, subtree)
            map(helper, subtree.children)
        elseif isleaf(subtree)
            push!(nodes, subtree)
        end
    end
end

let
    labels = []
    global spans
    function spans(t::SentimentTree,label,dep)
        empty!(labels)
        helper(t,label,dep)
        return labels
    end

    function helper(subtree,label,dep)
        if label
            if !isleaf(subtree)
                dep==true?push!(labels, subtree.label):nothing
                map(t->helper(t,label,dep), subtree.children)
            elseif isleaf(subtree)
                push!(labels, subtree.label)
            end
        else
            if !isleaf(subtree)
                dep==true?push!(labels, subtree.label):nothing
                map(t->helper(t,label,dep), subtree.children)
            elseif isleaf(subtree)
                push!(labels, subtree.data)
            end
        end
    end
end



function load_treebank_data(dep)
    dep==false?extract_file():nothing
    data = map(s->build_treebank_data(s,dep),TREEBANK_SPLITS )
end
function build_treebank_data(split,dep)
    data = dep==false?read_file(split*".txt"):read_file("SST_dep_parse_"*split*".txt")
end

function extract_file()
    download(TREEBANK_URL,"treebank.zip")
    r = ZipFile.Reader("treebank.zip");
    for f in r.files
        _, this_file = splitdir(f.name)
        split, _ = splitext(this_file)
        if split in TREEBANK_SPLITS
            lines = readlines(f)
            text = join(lines, "\n")
            file = split*".txt"
            open(file, "w") do f
                write(f, text)
            end
        end
    end
    close(r)
end

 function read_file(file)
        data = open(file, "r")
        lines=readlines(data)
        file=="train.txt"?deleteat!(lines,[2874,4341,5798,7408,8010]):nothing
        map(parse_line,lines)
    end

    function parse_line(line)
        ln = replace(line, "\n", "")
        tokens = tokenize_sexpr(ln)
        shift!(tokens)
        return within_bracket(tokens)[1]
    end

    function tokenize_sexpr(sexpr)
        tokker = r" +|[()]|[^ ()]+"
        filter(t -> t != " ", matchall(tokker, sexpr))
    end

    function within_bracket(tokens, state=1)
        (label, state) = next(tokens, state)
        children = []
        while !done(tokens, state)
            (token, state) = next(tokens, state)
            if token == "("
                (child, state) = within_bracket(tokens, state)
                push!(children, child)
            elseif token == ")"
                return SentimentTree(label, children), state
            else
                push!(children, SentimentTree(token))
            end
        end
    end

    function build_treebank_vocabs(trn,dev,tst)
        words = Set()
        labels = map(t->string(t),range(0,5))
        splits=[trn,dev,tst]
        w2g=Dict()
        for s in splits
            for i in 1:length(s)
                tree=s[i]
                push!(words, map(t->lowercase(t.label), nonterms(tree))...)
            end
        end
        words=unique(words)
        if isfile("SST_glove.jld")
            w2g=load("SST_glove.jld","w2g")
        else
            f="glove.840B.300d.txt"
            lines = readlines(f)
            for line in lines
                ln=split(line)
                if lowercase(ln[1]) in words
                    embedding=[parse(Float32,ln[i]) for i in 2:length(ln)  ]
                    w2g[lowercase(ln[1])]=embedding
                end
            end
                save("SST_glove.jld","w2g",w2g)
        end
        w2i, i2w = build_vocab(words)
        l2i, i2l = build_vocab(labels)
        return l2i, w2i, i2l, i2w, w2g
    end

    function build_vocab(xs)
        x2i = Dict(); i2x = Dict()
        for (i,x) in enumerate(xs)
            x2i[x] = i
            i2x[i] = x
        end
        return x2i, i2x
    end

function make_data!(deptrees,constrees, w2i, l2i,w2g)
    n=0
    for i in 1:length(deptrees)
        tree=deptrees[i]
        for nonterm in nonterms(tree)
            nonterm.data = get(w2g, nonterm.label, randn(Float32,(300,1)))
            for nt in nonterms(constrees[i])
                if sort(spans(nt,true,false))==sort(spans(tree,true,true))
                    nonterm.tag=get(l2i,nt.label,10)
                    n+=1
                else
                    nonterm.tag=10
                end
            end
        end
    end
    println("number of coincided nodes= ",n)
end

function binarized(trn)
    data=Any[]
    for sent in trn
        sent.tag!=3?push!(data,sent):nothing
    end
    return data
end
