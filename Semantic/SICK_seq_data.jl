for p in ("ZipFile","JLD","HDF5")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
using ZipFile
using JLD

const SPLIT=["train","trial","test_annotated"]
const nonglove=randn(Float32,(300,1))


function extract_file()
    for s in SPLIT
        download("http://alt.qcri.org/semeval2014/task1/data/uploads/sick_"*s*".zip","SICK_"*s*".zip")
        r=ZipFile.Reader("SICK_"*s*".zip")
        for f in r.files
            _, this_file = splitdir(f.name)
            name, _ = splitext(this_file)
            if name=="SICK_"*s
                lines = readlines(f)
                text = join(lines, "\n")
                file = "SICK_"*s*".txt"
                open(file, "w") do f
                    write(f, text)
                end
            end
        end
        close(r)
    end
end



function makedata(s)
    lines=readlines("SICK_"*s*".txt")
    lines=lines[2:end]
    data=Any[]
    for line in lines
        line=lowercase(line)
        s=split(line,"\t")
        r=parse(Float64,s[4])
        s[2]=replace(s[2],"n't"," not")
        s[2]=replace(s[2],"'s"," 's")
        s[2]=replace(s[2],","," ,")
        s[2]=replace(s[2],"."," .")
        s[2]=lowercase(s[2])
        s[3]=replace(s[3],"n't"," not")
        s[3]=replace(s[3],"'s"," 's")
        s[3]=replace(s[3],","," ,")
        s[3]=replace(s[3],"."," .")
        s[3]=lowercase(s[3])
        a=split(s[2]," ")
        b=split(s[3]," ")
        push!(data,[r,a,b])
    end
    return data
end


function build_dict(trn,dev,tst)
    pairs=[trn...,dev...,tst...]
    sentences=Any[]
    for pair in pairs
        push!(sentences,pair[2],pair[3])
    end
    words=Any[]
    for i in 1:length(sentences)
        for j in 1:length(sentences[i])
            push!(words,sentences[i][j])
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
        save("SICK_glove.jld","w2g",w2g)
    end
    return w2g,words
end

function make_data!(w2g,trn)
    data=Any[]
    for i in 1:length(trn)
        sent1=Any[]
        sent2=Any[]
        for j in 1:length(trn[i][2])
            push!(sent1,get(w2g,lowercase(trn[i][2][j]),randn(Float32,(300,1))))
        end
        for j in 1:length(trn[i][3])
            push!(sent2,get(w2g,lowercase(trn[i][3][j]),randn(Float32,(300,1))))
        end
        push!(data,[trn[i][1],sent1,sent2])
    end
    return data
end

function load_sick_data()
    extract_file()
    data = map(s->makedata(s),SPLIT )
end

#trn,dev,tst=load_sick_data()
#w2g,words=build_dict(trn,dev,tst)
#trn=build_data(w2g,trn)
#dev=build_data(w2g,dev)
#tst=build_data(w2g,tst)
