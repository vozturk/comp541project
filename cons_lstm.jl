Pkg.installed("ArgParse") == nothing && Pkg.add("ArgParse")

module Constituency
using Knet
using ArgParse

include("treebank_reader.jl")


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



function initweights(atype,nwords,embed,class,hsize,winit=0.1)
    w = Array{Any}(9)
    we(d...)=atype(winit.*randn(d...))
    bi(d...)=atype(zeros(d...))
    w[1]=we(3*hsize,embed) #weight of inputs -no forget gate weight
    w[2]=bi(3*hsize,1) #bias of gates
    w[3]=we(3*hsize,2*hsize) #hidden weights of i,o,u gates
    w[4]=we(hsize,hsize) #f1 hidden weight
    w[5]=we(hsize,hsize) #f2 hidden weight
    w[6]=bi(hsize,1) #bias of f
    w[7]=we(class,hsize) #weight of softmax
    w[8]=bi(class,1) #bias of softmax
    w[9]=we(embed,nwords) #embedding matrices
    return w
end

function lstm_leaf(w,data,p)
    hsize=size(w[4],2)
    x=w[end][:,data]
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

function lstm(w,h1,h2,c1,c2,p)
    hsize=size(w[4],2)
    gh = w[3]*vcat(h1,h2) .+ w[2]
    i = sigm.(gh[1:hsize,:])
    o = sigm.(gh[hsize+1:2*hsize,:])
    u = tanh.(gh[2*hsize+1:3*hsize,:])
    f1 = sigm.(w[4]*h2 .+ w[6])
    f2 = sigm.(w[5]*h1 .+ w[6])
    u=dropout(u,p).*(1-p)
    c = i.*u + f1.*c1 + f2.*c2
    h = o.*tanh.(c)
    return h,c
end


function helper(t,w,p,h=Any[],y=Any[])
    hl=nothing
    cl=nothing
    if length(t.children) == 1 && isleaf(t.children[1])
        l = t.children[1]
        hl,cl = lstm_leaf(w,l.data,p)
    elseif length(t.children) == 2
        t1,t2 = t.children[1], t.children[2]
        h1,c1,h,y=helper(t1,w,p,h,y)
        h2,c2,h,y=helper(t2,w,p,h,y)
        hl,cl = lstm(w,h1,h2,c1,c2,p)
    else
        error("invalid tree")
    end
    return (hl,cl,[h...,hl],[y...,t.data])
end

function predict(w,x,p)
    ypred=Any[]
    ygold=Any[]

         (_,_,h,y) = helper(x,w,p)
         ys = w[7]*hcat(h...) .+ w[8]
         push!(ypred,ys)
         push!(ygold,y)
     ypred= hcat(ypred...)
     ygold=vcat(ygold...)
     return ypred,ygold
end

function loss(w, x,p, lambda)
    ssq_par=0.0
    for i in 1:7
        ssq_par += sum(abs.(w[i]).^2)
    end
    ypred,ygold=predict(w,x,p)
     J=nll(ypred,ygold) + lambda*ssq_par/2.0
     return J
end

lossgradient=grad(loss)

function train(w,x,p,lambda,opt)
    g=lossgradient(w,x,p,lambda)
    update!(w,g,opt)
end


function Accuracy(w, data,p,lambda)
    ncorrect = 0.0
    nsentences = 0.0
    J=0.0
    for x in data
        y,ygold=predict(w,x,p)
        J += loss(w,x,p,lambda)
        ypred=convert(Array,y[:,end])
        for i in 1:size(ypred,2)
            ncorrect+= indmax(ypred[:,i])==ygold[end]?1.0:0.0
        end
    end
    nsentences=length(data)
    tag_acc=ncorrect/nsentences
    tag_loss=J/nsentences

    return tag_acc,tag_loss
end

function main(args)
    s = ArgParseSettings()
    s.description = "Constituency Tree LSTM"

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--dropout"; arg_type=Float64; default=0.5; help="dropout ratio")
        ("--lambda"; arg_type=Float32; default=0.0001f0; help="L2 regularization strength")
        ("--LR"; arg_type=Float64; default=0.05; help="Learning Rate")
        ("--embed"; arg_type=Int; default=300; help="word embedding size")
        ("--hidden"; arg_type=Int; default=150; help="LSTM hidden size")
        ("--epochs"; arg_type=Int; default=200; help="number of training epochs")
        ("--report"; arg_type=Int; default=500; help="report period in iters")
        ("--valid"; arg_type=Int; default=10000; help="valid period in iters")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--batchsize"; arg_type=Int; default=150; help="batchsize")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && Knet.setseed(o[:seed])
    atype = o[:atype] = !o[:usegpu] ? Array{Float32} : KnetArray{Float32}

# Data Loading
    trn,dev,tst=load_treebank_data()
    UNK="_UNK_"
    l2i,w2i,i2l,i2w = build_treebank_vocabs(trn)
    make_data!(trn,w2i,l2i)
    make_data!(dev,w2i,l2i)
    make_data!(tst,w2i,l2i)

# -----------------------------------------------------------------------

    nwords, ntags = length(w2i), length(l2i)
    model=optim=nothing; knetgc()
    model = initweights(atype,nwords,o[:embed],ntags,o[:hidden])
    optim=optimizers(model,Adagrad,lr=o[:LR])
    nw=0
    for x in model
        nw+=size(x,1)*size(x,2)
    end
    println("number of parameters: ",nw)



     for i in 1:o[:epochs]
         shuffle!(trn)

    @time for x in trn
             train(model,x,o[:dropout],o[:lambda],optim)
        end
        trnacc,trnloss = Accuracy(model, trn,o[:dropout],o[:lambda])
        devacc,devloss = Accuracy(model, dev,o[:dropout],o[:lambda])
        tstacc,tstloss = Accuracy(model, tst,o[:dropout],o[:lambda])
        println("EPOCH: ",i)
        @printf("trnacc=%.4f,   trnloss=%f\n", trnacc,trnloss)
        @printf("devacc=%.4f,   devloss=%f\n", devacc,devloss)
        @printf("tstacc=%.4f,   tstloss=%f\n", tstacc,tstloss)
    end

end

splitdir(PROGRAM_FILE)[end] == "cons_lstm.jl" && main(ARGS)
end
