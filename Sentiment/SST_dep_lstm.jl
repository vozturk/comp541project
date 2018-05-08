Pkg.installed("ArgParse") == nothing && Pkg.add("ArgParse")

module Dependency_SST
using Knet
using ArgParse
using JLD

include("SST_dep_parse_reader.jl")

function embedding(i2w,w2g)
    w2v=[]
    for i in 1:length(i2w)
        word=get(i2w,i,UNK)
        g=get(w2g,word,nonglove)
        push!(w2v,g)
    end
    return hcat(w2v...)
end


function minibatch(data,batchsize)
    batches=Any[]
    shuffle!(data)
    for i in 1:batchsize:length(data)
        if i+batchsize-1 > length(data)
            batch=data[i:end]
        else
            batch=data[i:i+batchsize-1]
        end
        push!(batches,batch)
    end
    return batches
end



function initweights(atype,w2v,embed,class,hsize,winit=0.1)
    w = Array{Any}(9)
    we(d...)=atype(winit.*randn(d...))
    bi(d...)=atype(zeros(d...))
    w[1]=we(4*hsize,embed) #weight of inputs
    w[2]=bi(4*hsize,1) #bias of gates
    w[3]=we(4*hsize,hsize) #hidden weights of i,o,u,f gates
    w[4]=we(hsize,hsize)
    w[5]=we(hsize,hsize)
    w[6]=bi(hsize,1)
    w[7]=we(class,hsize)
    w[8]=bi(class,1) #bias of softmax
    w[9]=convert(atype,w2v)
    return w
end

function lstm_leaf(w,x,p)
    hsize=size(w[4],2)
    x=w[9][:,x]
    x=reshape(x,length(x),1)
    gx = w[1]*x .+ w[2]
    i = sigm.(gx[1:hsize,:])
    o = sigm.(gx[hsize+1:2*hsize,:])
    u = tanh.(gx[2*hsize+1:3*hsize,:])
    u=dropout(u,p).*(1-p)
    c = i.*u
    h = o .* tanh.(c)
    return h,c
end

function lstm(w,x,hc,cc,p)
    hsize=size(w[4],2)
    n=length(hc)
    x=w[9][:,x]
    x=reshape(x,length(x),1)
    gh = w[1][1:3*hsize,:]*x + w[3][1:3*hsize,:]*sum(hc) + w[2][1:3*hsize,:]
    i = sigm.(gh[1:hsize,:])
    o = sigm.(gh[hsize+1:2*hsize,:])
    u = tanh.(gh[2*hsize+1:3*hsize,:])
    f=[w[1][3*hsize+1:end,:]*x + w[3][3*hsize+1:end,:]*h + w[2][3*hsize+1:end,:] for h in hc]
    u=dropout(u,p).*(1-p)
    c = i.*u + sum([sigm.(f[i]).*cc[i] for i in 1:length(cc)])
    h = o.*tanh.(c)
    return h,c
end


function helper(t,w,p,h=Any[],y=Any[])
    hl=nothing
    cl=nothing
    if isleaf(t)
        hl,cl = lstm_leaf(w,t.data,p)
    else
        hc=Any[]
        cc=Any[]
        for t1 in t.children
            h1,c1,h,y=helper(t1,w,p,h,y)
            push!(hc,h1)
            push!(cc,c1)
        end
        hl,cl = lstm(w,t.data,hc,cc,p)
    end
    return (hl,cl,[h...,hl],[y...,t.tag])
end

function predict(w,x,p)
    (_,_,h,ygold) = helper(x,w,p)
    ypred = w[7]*hcat(h...) .+ w[8]
    n=find(ygold)
    return ypred[:,Int[n...]],ygold[[n...]]
end

function loss(w,batch,p,lambda)
    ssq_par=0.0
    for i in 1:6
        ssq_par += sum(abs2.(w[i]))
    end
    yhat=[]
    y=[]
    for x in batch
        ypred,ygold=predict(w,x,p)
        push!(yhat,ypred)
        push!(y,ygold)
    end
    J=nll(hcat(yhat...),vcat(y...)) + lambda*ssq_par/2.0
    return J
end

lossgradient=grad(loss)

function train(w,x,p,lambda,opt,opt2)
    g=lossgradient(w,x,p,lambda)
    update!(w,g,[opt...,opt2])
end


function Accuracy(w, data,p,lambda,binary)
    ncorrect = 0.0
    nsentences = 0.0
    J=0.0
    for batch in data
        for x in batch
            y,ygold=predict(w,x,p,rand)
            ypred=convert(Array,y[:,end])
            if binary
                for i in 1:size(ypred,2)
                    ncorrect+= indmax(ypred[:,i])==1 && (ygold[end]==1 || ygold[end]==2)?1.0:0.0
                    ncorrect+= indmax(ypred[:,i])==2 && (ygold[end]==1 || ygold[end]==2)?1.0:0.0
                    ncorrect+= indmax(ypred[:,i])==4 && (ygold[end]==4 || ygold[end]==5)?1.0:0.0
                    ncorrect+= indmax(ypred[:,i])==5 && (ygold[end]==4 || ygold[end]==5)?1.0:0.0
                end
            else
                for i in 1:size(ypred,2)
                    ncorrect+= indmax(ypred[:,i])==ygold[end]?1.0:0.0
                end
            end
        end
        J += loss(w,batch,p,lambda,rand)
        nsentences+=length(batch)
    end
    tag_acc=ncorrect/nsentences
    tag_loss=J/nsentences

    return tag_acc,tag_loss
end

function main(args)
    t00=now();
    s = ArgParseSettings()
    s.description = "Constituency Tree LSTM"

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--binary"; action=:store_true; help="binary or fine-grained")
        ("--dropout"; arg_type=Float64; default=0.5; help="dropout ratio")
        ("--lambda"; arg_type=Float64; default=0.0001; help="L2 regularization strength")
        ("--LR"; arg_type=Float64; default=0.05; help="Learning Rate")
        ("--embed"; arg_type=Int; default=300; help="word embedding size")
        ("--hidden"; arg_type=Int; default=168; help="LSTM hidden size")
        ("--epochs"; arg_type=Int; default=100; help="number of training epochs")
        ("--report"; arg_type=Int; default=500; help="report period in iters")
        ("--valid"; arg_type=Int; default=10000; help="valid period in iters")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--batchsize"; arg_type=Int; default=10; help="batchsize")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && Knet.setseed(o[:seed])
    atype = o[:atype] = !o[:usegpu] ? Array{Float32} : KnetArray{Float32}

# Data Loading
    trn,dev,tst=load_treebank_data(true)
    trnc,devc,tstc=load_treebank_data(false)

    l2i,w2i,i2l,i2w,w2g = build_treebank_vocabs(trn,dev,tst)
    println("number of words not in glove is",length(w2i)-length(w2g))

    make_data!(trn,trnc,w2i,l2i)
    make_data!(dev,devc,w2i,l2i)
    make_data!(tst,tstc,w2i,l2i)

    if o[:binary]
        trn=binarized(trn)
        dev=binarized(dev)
        tst=binarized(tst)
    end


    trn=minibatch(trn,o[:batchsize])
    dev=minibatch(dev,o[:batchsize])
    tst=minibatch(tst,o[:batchsize])



# -----------------------------------------------------------------------

    nwords, ntags = length(w2i), length(l2i)
    w2v=embedding(i2w,w2g)
    model=optim=nothing; knetgc()
    model = initweights(atype,w2v,o[:embed],ntags,o[:hidden])
    optim=optimizers(model[1:end-1],Adagrad,lr=o[:LR])
    optim2=optimizers(model[end],Adagrad,lr=0.1)

    println("startup time: ", Int((now()-t00).value)*0.001); flush(STDOUT)

     for i in 1:o[:epochs]
    @time for x in trn
             train(model,x,o[:dropout],o[:lambda],optim,optim2)
        end
        trnacc,trnloss = Accuracy(model, trn,o[:dropout],o[:lambda],o[:binary])
        devacc,devloss = Accuracy(model, dev,o[:dropout],o[:lambda],o[:binary])
        tstacc,tstloss = Accuracy(model, tst,o[:dropout],o[:lambda],o[:binary])
        println("EPOCH: ",i)
        @printf("trnacc=%.4f,   trnloss=%f\n", trnacc,trnloss)
        @printf("devacc=%.4f,   devloss=%f\n", devacc,devloss)
        @printf("tstacc=%.4f,   tstloss=%f\n", tstacc,tstloss)
    end

end

splitdir(PROGRAM_FILE)[end] == "SST_dep_lstm.jl" && main(ARGS)
end
