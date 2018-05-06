for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
module Semantic_LSTM
using ArgParse
using Knet

include("SICK_seq_data.jl")

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

function initweights(atype,dir,lay,pdrop,tags,embed,winit=0.01)
    w = Array{Any}(7)
    x=dir==false?1:2
    hidden=lay==1?150:108
    we(d...)=atype(winit.*randn(d...))
    bi(d...)=atype(zeros(d...))
    w[2]=we(x*hidden,x*hidden)
    w[3]=we(x*hidden,x*hidden)
    w[4]=bi(x*hidden,1)
    w[5]=we(tags,x*hidden)
    w[6]=bi(tags,1)
    w[7],w[1]=rnninit(embed,hidden;bidirectional=dir,rnnType=:lstm,numLayers=lay,dropout=pdrop)
    return w
end

function sentence_rep(w,pair) #data is pair
    x1=hcat(pair[2]...)
    x1=convert(KnetArray{Float32},x1)
    (y1,_) = rnnforw(w[7],w[1],x1)
    x2 = hcat(pair[3]...)
    x2=convert(KnetArray{Float32},x2)
    (y2,_) = rnnforw(w[7],w[1],x2)
    rep=[pair[1],y1[:,end],y2[:,end]]
    return rep
end

function relatedness(w,data,tags)
    rep=sentence_rep(w,data)
    h1=rep[2].*rep[3]
    h1=reshape(h1,length(h1),1)
    h2=abs.(rep[2]-rep[3])
    h2=reshape(h2,length(h2),1)
    hs=sigm.(w[2]*h1 + w[3]*h2 + w[4])
    p=exp.(logp(w[5]*hs + w[6]))
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

function loss(w,batch,tags,lambda)
    J=0.0;
    for data in batch
        y,p=relatedness(w,data,tags)
        pgold=sparse_prob(data[1],tags)
        div=KL_div(pgold,p)
        J+=div
    end
    l=J/length(batch) + lambda*sum(abs2.(w[1]))/2.0
    return l
end

lossgradient=grad(loss)

function train(w,x,tags,lambda,opt)
    g=lossgradient(w,x,tags,lambda)
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



function metrics(w,data,tags,lambda) #data is whole trn or dev or tst
    ypred=Any[]
    ygold=Any[]
    J=0.0
    npair=0.0
    for batch in data
        for pair in batch
            yhat,_=relatedness(w,pair,tags)
            y=pair[1]
            push!(ypred,yhat)
            push!(ygold,y)
        end
        npair+=length(batch)
        J+=loss(w,batch,tags,lambda)
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
    s.description = "Baseline LSTM"

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--bilstm"; action=:store_true; help="use bidirectional or not")
        ("--layer"; arg_type=Int; default=1; help="number of layers")
        ("--dropout"; arg_type=Float64; default=0.5; help="dropout ratio")
        ("--LR"; arg_type=Float64; default=0.05; help="learning rate")
        ("--lambda"; arg_type=Float64; default=0.001; help="L2 regularization strength")
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
    trn=make_data!(w2g,trn)
    dev=make_data!(w2g,dev)
    tst=make_data!(w2g,tst)

    trn=minibatch(trn,o[:batchsize])
    dev=minibatch(dev,o[:batchsize])
    tst=minibatch(tst,o[:batchsize])


    #println(trn[2])


    # build model
    w=optim=nothing
    w = initweights(atype, o[:bilstm],o[:layer],o[:dropout],o[:tags], o[:embed])
    optim=optimizers(w,Adagrad,lr=o[:LR])

    println("startup time: ", Int((now()-t00).value)*0.001); flush(STDOUT)


    for i in 1:o[:epochs]
        shuffle!(trn)
        @time for pair in trn
            train(w,pair,o[:tags],o[:lambda],optim)
        end
        trnr,trnrho,trnmse,trnloss = metrics(w, trn,o[:tags],o[:lambda])
        devr,devrho,devmse,devloss = metrics(w, dev,o[:tags],o[:lambda])
        tstr,tstrho,tstmse,tstloss = metrics(w, tst,o[:tags],o[:lambda])
        println("EPOCH: ",i)
        @printf("Train: Pearson's r=%f,   Spearman's Rho=%f,    MSE=%f,   loss=%f\n",trnr,trnrho,trnmse,trnloss)
        @printf("Dev: Pearson's r=%f,     Spearman's Rho=%f,    MSE=%f,   Loss=%f\n",devr,devrho,devmse,devloss)
        @printf("Test: Pearson's r=%f,    Spearman's Rho=%f,    MSE=%f,   loss=%f\n",tstr,tstrho,tstmse,tstloss)
    end


end
splitdir(PROGRAM_FILE)[end] == "SICK_lstm.jl" && main(ARGS)
end
