for p in ("JLD","HDF5")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
using ZipFile
using JLD
const SICK_SPLITS = ("train","trial","test_annotated")
const nonglove=randn(Float32,(300,1))

mutable struct SentimentTree
    label
    children
    data

    SentimentTree(x) = new(x,nothing,nothing)
    SentimentTree(x,y) = new(x,y,nothing)
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
    function spans(t::SentimentTree)
        empty!(labels)
        helper(t)
        return labels
    end

    function helper(subtree)
        if !isleaf(subtree)
            push!(labels, subtree.label)
            map(helper, subtree.children)
        elseif isleaf(subtree)
            push!(labels, subtree.label)
        end
    end
end


function load_sick_data()
    data = map(s->build_sick_data(s),SICK_SPLITS )
end
function build_sick_data(split)
    data=Any[]
    trees = read_file("SICK_dep_parse_"*split*".txt")
    for i in 1:2:length(trees)
        push!(data,[trees[i],trees[i+1]])
    end
    return data
end

 function read_file(file)
        data = open(file, "r") do f
            map(parse_line, readlines(f))
        end
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

    function build_dict(trn,dev,tst)
        words = Set()
        splits=[trn,dev,tst]
        for s in splits
            for i in 1:length(s)
                tree1=s[i][1]
                tree2=s[i][2]
                push!(words, map(t->lowercase(t), spans(tree1))...)
                push!(words, map(t->lowercase(t), spans(tree2))...)
            end
        end
        words=unique(words)
        if isfile("SICK_glove.jld")
                w2g=load("SICK_glove.jld","w2g")
        else
            f="glove.840B.300d.txt"
            lines = readlines(f)
            w2g=Dict()
            for line in lines
                ln=split(line)
                if lowercase(ln[1]) in words
                    embedding=[parse(Float32,ln[i]) for i in 2:length(ln)  ]
                    w2g[lowercase(ln[1])]=embedding
                end
            end
        end
        return w2g,words
    end


function make_data!(trn, w2g)
    for pair in trn
        for tree in pair
            for nt in nonterms(tree)
                ind = get(w2g, lowercase(nt.label), randn(Float32,(300,1)))
                nt.data = ind
            end
            tree.data=parse(Float64,tree.label)
        end
    end
end
