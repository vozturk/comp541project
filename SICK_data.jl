for p in ("ZipFile",)
    Pkg.installed(p) == nothing && Pkg.add(p)
end
using ZipFile

function extract_file(s)
    download("http://alt.qcri.org/semeval2014/task1/data/uploads/sick_"*s*".zip","sick_"*s*".zip")
    r=ZipFile.Reader("sick_"*s*".zip")
    for f in r.files
        name,t=splitext(f.name)
        if name=="SICK_"*s
            lines = readlines(f)
            text = join(lines, "\n")
            file = "sick_"*s*".txt"
            open(file, "w") do f
                write(f, text)
            end
        end
    end
    close(r)
end


function makedata(file)
    lines=readlines(file)
    lines=lines[2:end]
    ID=Any[]
    A=Any[]
    B=Any[]
    Rate=Any[]
    for i in 1:length(lines)
        line=lines[i]
        line=lowercase(line)
        s=split(line,"\t")
        id=parse(Int,s[1])
        push!(ID,id)
        r=parse(Float64,s[4])
        push!(Rate,r)
        s[2]=replace(s[2],"n't"," n't")
        s[2]=replace(s[2],"'s"," 's")
        s[2]=replace(s[2],","," ,")
        s[2]=replace(s[2],"."," .")
        s[3]=replace(s[3],"n't"," n't")
        s[3]=replace(s[3],"'s"," 's")
        s[3]=replace(s[3],","," ,")
        s[3]=replace(s[3],"."," .")
        a=split(s[2]," ")
        b=split(s[3]," ")
        push!(A,a)
        push!(B,b)
    end
    return [ID,A,B,Rate]
end

function build_dict(trn,dev,tst)
all=[trn[2]...,dev[2]...,tst[2]...,trn[3]...,dev[3]...,tst[3]...]
words=Any[]
for i in 1:length(all)
    for j in 1:length(all[i])
    push!(words,all[i][j])
end
end
words=unique(words)
w2i=Dict()
i2w=Dict()
for i in 1:length(words)
    x=words[i]
    w2i[x]=i
    i2w[i]=x
end
i=length(words)+1
w2i["UNK"]=i
i2w[i]="UNK"
return w2i,i2w,words
end

function build_data(w2i,trn)
    data=Any[]
    data1=Any[]
    for i in [2,3]
        a=Any[]
        for j in 1:length(trn[i])
            h=Any[]
            for k in 1:length(trn[i][j])
                push!(h,get(w2i,trn[i][j][k],w2i["UNK"]))
            end
            push!(a,h)
        end
        push!(data1,a)
    end
    data=[trn[1],data1[1],data1[2],trn[4]]
    return data
end


function load_sick_data()
SPLIT=["train","trial","test_annotated"]
data=Any[]
for s in SPLIT
extract_file(s)
push!(data,makedata("sick_"*s*".txt"))
end
return data
end
#trn,dev,tst=load_sick_data()
#w2i,i2w,words=build_dict(trn,dev,tst)
#trn=build_data(w2i,trn)
#dev=build_data(w2i,dev)
#tst=build_data(w2i,tst)
