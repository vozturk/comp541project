Pkg.installed("ArgParse") == nothing && Pkg.add("ArgParse")

module Constituency_sick
using Knet
using ArgParse

include("SICK_cons_parse_reader.jl")


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



function initweights(atype,embed,class,hsize,winit=0.1)
    w = Array{Any}(11)
    we(d...)=atype(winit.*randn(d...))
    bi(d...)=atype(zeros(d...))
    w[1]=we(3*hsize,embed) #weight of inputs -no forget gate weight
    w[2]=bi(3*hsize,1) #bias of gates
    w[3]=we(3*hsize,2*hsize) #hidden weights of i,o,u gates
    w[4]=we(hsize,hsize) #f1 hidden weight
    w[5]=we(hsize,hsize) #f2 hidden weight
    w[6]=bi(hsize,1) #bias of f
    w[7]=we(hsize,hsize)
    w[8]=we(hsize,hsize)
    w[9]=bi(hsize,1)
    w[10]=we(class,hsize)
    w[11]=bi(class,1) #bias of softmax
    return w
end

function lstm_leaf(w,x,p)
    hsize=size(w[4],2)
    x=convert(KnetArray{Float32},x)
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


function sentence_rep(t,w,p,h=Any[])
    hl=nothing
    cl=nothing
    if length(t.children) == 1 && isleaf(t.children[1])
        l = t.children[1]
        hl,cl = lstm_leaf(w,l.data,p)
    elseif length(t.children) == 2
        t1,t2 = t.children[1], t.children[2]
        h1,c1,h=sentence_rep(t1,w,p,h)
        h2,c2,h=sentence_rep(t2,w,p,h)
        hl,cl = lstm(w,h1,h2,c1,c2,p)
    else
        error("invalid tree")
    end
    return (hl,cl,[h...,hl])
end

function relatedness(w,pair,p,tags)
    (_,_,hl) = sentence_rep(pair[1].children[1],w,p)
    (_,_,hr) = sentence_rep(pair[2].children[1],w,p)
    h1=hl[end].*hr[end]
    h2=abs.(hl[end]-hr[end])
    hs=sigm.(w[7]*h1 + w[8]*h2 + w[9])
    p=exp.(logp(w[10]*hs + w[11]))
    reshape(p,size(p,1),1)
    r=convert(KnetArray{Float32},range(1,tags))
    y=sum(r.*p)
    return y,p
end

function sparse_prob(x,tags)
    p=zeros(1,tags)
    for j in 1:tags
        if j==floor(x)+1
            p[j]=x-floor(x)
        elseif j==floor(x)
            p[j]=floor(x)-x+1
        end
    end
    return p
end

function KL_div(p,q)
    n=length(p)
    sum=0
    for i in 1:n
        sum+= p[i]<=0 || q[i]<=0 ? 0.0 : p[i]* log(p[i]/q[i])
    end
    return sum
end

function loss(w,batch,prob,tags,lambda)
    J=0.0
    for pair in batch
        y,p=relatedness(w,pair,prob,tags)
        if pair[1].data==pair[2].data
            pgold=sparse_prob(pair[1].data,tags)
        else print("it is not a pair")
        end
        div=KL_div(pgold,p)
        J+=div
    end
    ssq_par=0.0
    for i in 1:6
        ssq_par += sum(abs2.(w[i]))
    end
    J=J/length(batch) + lambda*ssq_par/2.0
    return J
end

lossgradient=grad(loss)

function train(w,x,prob,tags,lambda,opt)
    g=lossgradient(w,x,prob,tags,lambda)
    update!(w,g,opt)
end

function spearmans_rho(x,y)
    x1=sortperm(x)
    y1=sortperm(y)
    n=length(x)
    r1=zeros(n,1)
    for i in 1:n
        r=find(y1.==x1[i])
        r1[i]=r[1]
    end
    rho=1-((6*sum(abs2.(range(1,n)-r1)))/(n*(n^2-1)))
    return rho
end



function metrics(w,data,prob,tags,lambda) #data is whole trn or dev or tst
    ypred=Any[]
    ygold=Any[]
    J=0.0
    npair=0.0
    for batch in data
        for pair in batch
            yhat,_=relatedness(w,pair,prob,tags)
            y=pair[1].data
            push!(ypred,yhat)
            push!(ygold,y)
            npair+=1
        end
        J+=loss(w,batch,prob,tags,lambda)
    end
    r=cor(ypred,ygold)
    rho=spearmans_rho(ypred,ygold)
    mse=sum(abs2.(ypred-ygold))/npair
    avrgloss=J/length(data)
    return r,rho,mse,avrgloss
end



function main(args)
    t00=now()
    s = ArgParseSettings()
    s.description = "Constituency LSTM"

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--bilstm"; action=:store_true; help="use bidirectional or not")
        ("--layer"; arg_type=Int; default=1; help="number of layers")
        ("--dropout"; arg_type=Float64; default=0.5; help="dropout ratio")
        ("--LR"; arg_type=Float64; default=0.05; help="learning rate")
        ("--lambda"; arg_type=Float64; default=0.001; help="L2 regularization strength")
        ("--hidden"; arg_type=Int; default=142; help="hiddensize size")
        ("--embed"; arg_type=Int; default=300; help="word embedding size")
        ("--epochs"; arg_type=Int; default=100; help="number of training epochs")
        ("--report"; arg_type=Int; default=500; help="report period in iters")
        ("--valid"; arg_type=Int; default=10000; help="valid period in iters")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--batchsize"; arg_type=Int; default=20; help="batchsize")
        ("--tags"; arg_type=Int; default=5; help="number of tags")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && Knet.setseed(o[:seed])
    atype = o[:atype] = !o[:usegpu] ? Array{Float32} : KnetArray{Float32}


    trn,dev,tst=load_sick_data()
    println("SICK data is ready")
    w2g,words=build_dict(trn,dev,tst)
    println("word vectors dictionary is ready")
    println("number of words not in glove is",length(words)-length(w2g))
    make_data!(trn,w2g)
    make_data!(dev,w2g)
    make_data!(tst,w2g)

    trn=minibatch(trn,o[:batchsize])
    dev=minibatch(dev,o[:batchsize])
    tst=minibatch(tst,o[:batchsize])


    # build model
    w=optim=nothing
    w = initweights(atype,o[:embed],o[:tags],o[:hidden])
    optim=optimizers(w,Adagrad,lr=o[:LR])

    println("startup time: ", Int((now()-t00).value)*0.001); flush(STDOUT)


    for i in 1:o[:epochs]
        shuffle!(trn)
        @time for pair in trn
            train(w,pair,o[:dropout],o[:tags],o[:lambda],optim)
        end
        trnr,trnrho,trnmse,trnloss = metrics(w, trn,o[:dropout],o[:tags],o[:lambda])
        devr,devrho,devmse,devloss = metrics(w, dev,o[:dropout],o[:tags],o[:lambda])
        tstr,tstrho,tstmse,tstloss = metrics(w, tst,o[:dropout],o[:tags],o[:lambda])
        println("EPOCH: ",i)
        @printf("Train: Pearson's r=%f,   Spearman's Rho=%f,    MSE=%f,   loss=%f\n",trnr,trnrho,trnmse,trnloss)
        @printf("Dev: Pearson's r=%f,     Spearman's Rho=%f,    MSE=%f,   Loss=%f\n",devr,devrho,devmse,devloss)
        @printf("Test: Pearson's r=%f,    Spearman's Rho=%f,    MSE=%f,   loss=%f\n",tstr,tstrho,tstmse,tstloss)
    end


end
splitdir(PROGRAM_FILE)[end] == "SICK_cons_lstm.jl" && main(ARGS)
end
