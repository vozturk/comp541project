using Knet
Pkg.installed("ArgParse") == nothing && Pkg.add("ArgParse")
using ArgParse

function make_batches(data, batchsize) #data= trn or dev or tst
    batches = []
    for k = 1:batchsize:length(data[1])
        lo = k
        hi = min(k+batchsize-1, length(data[1]))
        samples = (data[1][lo:hi],data[2][lo:hi],data[3][lo:hi],data[4][lo:hi])
        push!(batches,samples)
    end
    return batches
end

function initweights(atype,dir,lay,pdrop,hidden,words,tags, embed,  winit=0.01)
    w = Array{Any}(8)
    x=dir==false?1:2
    we(d...)=atype(winit.*randn(d...))
    bi(d...)=atype(zeros(d...))
    w[2]=we(embed,words)
    w[3]=we(x*hidden,x*hidden)
    w[4]=we(x*hidden,x*hidden)
    w[5]=bi(x*hidden,1)
    w[6]=we(tags,x*hidden)
    w[7]=bi(tags,1)
    w[8],w[1]=rnninit(embed,hidden;bidirectional=dir,rnnType=:lstm,numLayers=lay,dropout=pdrop)
    return w
end

function sentence_rep(w,batch)
    rep=Any[]
    for i in 1:length(batch[4])
        h=Any[]
        x1=batch[2][i]
        x2=batch[3][i]
        x = w[2][:,x1[1]]
        for j in 2:length(x1)
            x=hcat(x,w[2][:,x1[j]])
        end
        (y,_) = rnnforw(w[8],w[1],x)
        push!(h,y[:,end])
        x = w[2][:,x2[1]]
        for j in 2:length(x2)
            x=hcat(x,w[2][:,x2[j]])
        end
        (y,_) = rnnforw(w[8],w[1],x)
        push!(h,y[:,end])
        push!(h,batch[4][i])
        push!(rep,h)
    end
    return rep
end

function relatedness(w,batch,tags)
    rep=sentence_rep(w,batch)
    prob=Any[]
    ypred=Any[]
    for i in 1:length(rep)
        h=rep[i]
        h1=h[1].*h[2]
        h1=reshape(h1,length(h1),1)
        h2=abs.(h[1]-h[2])
        h2=reshape(h2,length(h2),1)
        hs=sigm.(w[3]*h1 + w[4]*h2 + w[5])
        p1=exp.(logp(w[6]*hs + w[7]))
        p=convert(Array{Float32},p1[:,1])
        y=sum(range(1,tags).*p)
        push!(prob,p)
        push!(ypred,y)
    end
    return ypred,prob
end

function sparse_prob(y,tags)
    prob=Any[]
    for x in y
        p=zeros(1,tags)
        for j in 1:tags
            if j==floor(x)+1
                p[j]=x-floor(x)
            elseif j==floor(x)
                p[j]=floor(x)-x+1
            end
        end
        push!(prob,p)
    end
    return prob
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
    y,p=relatedness(w,batch,tags)
    pgold=sparse_prob(batch[4],tags)
    div=0
    for i in 1:length(batch[4])
        div+=KL_div(pgold[i],p[i])
    end
    J=div/length(batch[4])
    return J
end

function spearmans_rho(x,y)
    x1=sortperm(x)
    y1=sortperm(y)
    r1=zeros(length(x),1)
    for i in 1:length(x1)
        r=find(y1.==x1[i])
        r1[i]=r[1]
    end
    return cor(range(1,length(x)),r1)
end



function metrics(w,data,tags,lambda)
    ypred=Any[]
    ygold=Any[]
    J=0.0f0
    npair=0.0f0
    for batch in data
        yhat,_=relatedness(w,batch,tags)
        y=batch[4]
        push!(ypred,yhat...)
        push!(ygold,y...)
        J+=loss(w,batch,tags,lambda)
        npair+=length(batch[4])
    end
    r=cor(ypred,ygold)
    rho=spearmans_rho(ypred,ygold)[1]
    mse=sum(abs2.(ypred-ygold))/npair
    avrgloss=J/npair
    return r,rho,mse,avrgloss
end



function main(trn,dev,tst,w2i,args)
    s = ArgParseSettings()
    s.description = "Baseline LSTM"

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--bidirectional"; arg_type=Bool; default=false; help="use bidirectional or not")
        ("--layer"; arg_type=Int; default=1; help="number of layers")
        ("--dropout"; arg_type=Float32; default=0.0f0; help="dropout ratio")
        ("--lambda"; arg_type=Float32; default=0.0001f0; help="L2 regularization strength")
        ("--embed"; arg_type=Int; default=300; help="word embedding size")
        ("--hidden"; arg_type=Int; default=150; help="LSTM hidden size")
        ("--epochs"; arg_type=Int; default=20; help="number of training epochs")
        ("--report"; arg_type=Int; default=500; help="report period in iters")
        ("--valid"; arg_type=Int; default=10000; help="valid period in iters")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--tags"; arg_type=Int; default=-5; help="number of tags")
        ("--batchsize"; arg_type=Int; default=25; help="batchsize")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && Knet.setseed(o[:seed])
    atype = o[:atype] = !o[:usegpu] ? Array{Float32} : KnetArray{Float32}

    tags=5;

    # build model
    nwords= length(w2i)
    w = initweights(
        atype, o[:bidirectional],o[:layer],o[:dropout],o[:hidden], nwords,tags, o[:embed])

    # make batches
    trn = make_batches(trn, o[:batchsize])
    dev = make_batches(dev, o[:batchsize])
    tst = make_batches(tst, o[:batchsize])


        # validation
    trnr,trnrho,trnmse,trnloss = metrics(w, trn,tags,o[:lambda])
    devr,devrho,devmse,devloss = metrics(w, dev,tags,o[:lambda])
    tstr,tstrho,tstmse,tstloss = metrics(w, tst,tags,o[:lambda])

        # report
    @printf("Train: Pearson's r=%f,   Spearman's Rho=%f,    MSE=%f,   loss=%f\n",trnr,trnrho,trnmse,trnloss)
    @printf("Dev: Pearson's r=%f,     Spearman's Rho=%f,    MSE=%f,   Loss=%f\n",devr,devrho,devmse,devloss)
    @printf("Test: Pearson's r=%f,    Spearman's Rho=%f,    MSE=%f,   loss=%f\n",tstr,tstrho,tstmse,tstloss)


end

#main(trn,dev,tst,w2i"--usegpu")
