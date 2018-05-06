Pkg.installed("ArgParse") == nothing && Pkg.add("ArgParse")

module Constituency
using Knet
using ArgParse

include("SST_cons_parse_reader.jl")


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

function lstm_leaf(w,data,p,rand)
    hsize=size(w[4],2)
    if rand
        x=w[end][:,data]
        x=reshape(x,length(x),1)
    else
        x=convert(KnetArray{Float32},data)
        x=reshape(x,length(x),1)
    end
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


function helper(t,w,p,rand,h=Any[],y=Any[])
    hl=nothing
    cl=nothing
    if length(t.children) == 1 && isleaf(t.children[1])
        l = t.children[1]
        hl,cl = lstm_leaf(w,l.data,p,rand)
    elseif length(t.children) == 2
        t1,t2 = t.children[1], t.children[2]
        h1,c1,h,y=helper(t1,w,p,rand,h,y)
        h2,c2,h,y=helper(t2,w,p,rand,h,y)
        hl,cl = lstm(w,h1,h2,c1,c2,p)
    else
        error("invalid tree")
    end
    return (hl,cl,[h...,hl],[y...,t.data])
end

function predict(w,x,p,rand)
    (_,_,h,ygold) = helper(x,w,p,rand)
    ypred = w[7]*hcat(h...) .+ w[8]
    return ypred,ygold
end

function loss(w,batch,p,lambda,rand)
    ssq_par=0.0
    for i in 1:6
        ssq_par += sum(abs2.(w[i]))
    end
    yhat=[]
    y=[]
    for x in batch
        ypred,ygold=predict(w,x,p,rand)
        push!(yhat,ypred)
        push!(y,ygold)
    end
     J=nll(hcat(yhat...),vcat(y...)) + lambda*ssq_par/2.0
     return J
end

lossgradient=grad(loss)

function train(w,x,p,lambda,opt,rand)
    g=lossgradient(w,x,p,lambda,rand)
    update!(w,g,opt)
end


function Accuracy(w, data,p,lambda,rand)
    ncorrectb = 0.0
    ncorrect=0.0
    nsentences = 0.0
    J=0.0
    for batch in data
        for x in batch
            y,ytrue=predict(w,x,p,rand)
            ypred=convert(Array,y[:,end])
            ygold=ytrue[end]
            ncorrectb+= indmax(ypred)==1 && (ygold==1 || ygold==2)?1.0:0.0
            ncorrectb+= indmax(ypred)==2 && (ygold==1 || ygold==2)?1.0:0.0
            ncorrectb+= indmax(ypred)==4 && (ygold==4 || ygold==5)?1.0:0.0
            ncorrectb+= indmax(ypred)==5 && (ygold==4 || ygold==5)?1.0:0.0
            ncorrect+= indmax(ypred)==ygold?1.0:0.0
        end
        J += loss(w,batch,p,lambda,rand)
        nsentences+=length(batch)
    end
    tag_acc=ncorrect/nsentences
    tag_acc_b=ncorrectb/nsentences
    tag_loss=J/length(data)

    return tag_acc,tag_acc_b,tag_loss
end

function main(args)
    t00=now();
    s = ArgParseSettings()
    s.description = "Constituency Tree LSTM"

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--random"; action=:store_true; help="randomized or glove word vectors")
        ("--dropout"; arg_type=Float64; default=0.5; help="dropout ratio")
        ("--lambda"; arg_type=Float64; default=0.0001; help="L2 regularization strength")
        ("--LR"; arg_type=Float64; default=0.05; help="Learning Rate")
        ("--embed"; arg_type=Int; default=300; help="word embedding size")
        ("--hidden"; arg_type=Int; default=150; help="LSTM hidden size")
        ("--epochs"; arg_type=Int; default=100; help="number of training epochs")
        ("--report"; arg_type=Int; default=500; help="report period in iters")
        ("--valid"; arg_type=Int; default=10000; help="valid period in iters")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--batchsize"; arg_type=Int; default=20; help="batchsize")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && Knet.setseed(o[:seed])
    atype = o[:atype] = !o[:usegpu] ? Array{Float32} : KnetArray{Float32}

# Data Loading
    trn,dev,tst=load_treebank_data()


    l2i,w2i,i2l,i2w,w2g,words = build_treebank_vocabs(trn,dev,tst)
    !o[:random]?println("number of words not in glove is",length(words)-length(w2g)):nothing

    make_data!(trn,w2i,l2i,w2g,o[:random])
    make_data!(dev,w2i,l2i,w2g,o[:random])
    make_data!(tst,w2i,l2i,w2g,o[:random])

    trn=minibatch(trn,o[:batchsize])
    dev=minibatch(dev,o[:batchsize])
    tst=minibatch(tst,o[:batchsize])


    println(length(trn))


# -----------------------------------------------------------------------

    nwords, ntags = length(w2i), length(l2i)
    model=optim=nothing; knetgc()
    model = initweights(atype,nwords,o[:embed],ntags,o[:hidden])
    optim=optimizers(model,Adagrad,lr=o[:LR])

    println("startup time: ", Int((now()-t00).value)*0.001); flush(STDOUT)

     for i in 1:o[:epochs]
         shuffle!(trn)

    @time for x in trn
             train(model,x,o[:dropout],o[:lambda],optim,o[:random])
        end
        trnacc,trnaccb,trnloss = Accuracy(model, trn,o[:dropout],o[:lambda],o[:random])
        devacc,devaccb,devloss = Accuracy(model, dev,o[:dropout],o[:lambda],o[:random])
        tstacc,tstaccb,tstloss = Accuracy(model, tst,o[:dropout],o[:lambda],o[:random])
        println("EPOCH: ",i)
        @printf("trnacc=%.4f,   trnaccbin=%.4f,   trnloss=%f\n", trnacc,trnaccb,trnloss)
        @printf("devacc=%.4f,   devaccbin=%.4f,   devloss=%f\n", devacc,devaccb,devloss)
        @printf("tstacc=%.4f,   tstaccbin=%.4f,   tstloss=%f\n", tstacc,tstaccb,tstloss)
    end

end

splitdir(PROGRAM_FILE)[end] == "SST_cons_lstm.jl" && main(ARGS)
end
