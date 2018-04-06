for p in ("ZipFile",)
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using ZipFile

const TREEBANK_URL = "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"
const TREEBANK_SPLITS = ("train","dev", "test")
const UNK="_UNK_"





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
        end
    end
end


function load_treebank_data()
    extract_file()
    data = map(s->build_treebank_data(s),TREEBANK_SPLITS )
end
function build_treebank_data(split)
    data = read_file(split*".txt")
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

    function build_treebank_vocabs(trn)
        words = Set()
        labels = Set()
            for i in 1:length(trn)
                tree=trn[i]
                push!(words, map(t->t.label, leaves(tree))...)
                push!(labels, map(t->t.label, nonterms(tree))...)
            end
        push!(words, UNK)
        words=unique(words)
        w2i, i2w = build_vocab(words)
        l2i, i2l = build_vocab(labels)
        return l2i, w2i, i2l, i2w
    end

    function build_vocab(xs)
        x2i = Dict(); i2x = Dict()
        for (i,x) in enumerate(xs)
            x2i[x] = i
            i2x[i] = x
        end
        return x2i, i2x
    end

function make_data!(trees::SentimentTrees, w2i, l2i)
    for tree in trees
        for leaf in leaves(tree)
            ind = get(w2i, leaf.label, w2i[UNK])
            leaf.data = ind
        end
        for nonterm in nonterms(tree)
            nonterm.data = l2i[nonterm.label]
        end
    end
end
