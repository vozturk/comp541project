for p in ("ArgParse","JLD","HDF5")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

module LSTM_sst
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

    #=function seq_data(trn)
        data=Any[]
        for x in trn
            ygold=Int[]
            ind=Int[]
            span=spans(x,true)
            n=length(span)
            a=[]
            for i in 1:n
                push!(a,span[1:i])
            end
            for i in 1:length(a)
                for nt in nonterms(x)
                    if sort(a[i])==sort(spans(nt,true)) && !isleaf(nt)
                        push!(ygold,nt.data)
                        push!(ind,i)
                    end
                end
            end
            push!(data,[x,ygold,ind])
        end
        return data
    end=#

    #=function binarize(trn)
        data=Any[]
        for x in trn
            if x[2][end]!=3
                push!(data,x)
            end
        end
        return data
    end=#




    function initweights(atype,dir,lay,pdrop, tags, embed, winit=0.01)
        w = Array{Any}(4)
        x=dir==false?1:2
        hidden=lay==1?168:120
        we(d...)=atype(winit.*randn(d...))
        bi(d...)=atype(zeros(d...))
        w[2]=we(tags,x*hidden)
        w[3]=bi(tags,1)
        w[4],w[1]=rnninit(embed,hidden;bidirectional=dir,rnnType=:lstm,numLayers=lay,dropout=pdrop)
        return w
    end

    function predict(ws, xs)
        x=hcat(spans(xs,false)...)
        x = convert(KnetArray{Float32},x)
        (y,_) = rnnforw(ws[4],ws[1],x)
        ypred=ws[2]*y .+ ws[3]
        #return ypred[:,xs[3]],xs[2]
        return ypred[:,end],xs.data
    end

    function loss(w, batch,lambda)
        yhat=[]
        y=[]
        for x in batch
            ypred,ygold=predict(w,x)
            push!(yhat,ypred)
            push!(y,ygold)
        end
         J=nll(hcat(yhat...),vcat(y...))
         #+ lambda*(sum(abs2.(w[1])))/2.0
         return J
    end

    lossgradient=grad(loss)

    function train(w,x,lambda,opt)
        g=lossgradient(w,x,lambda)
        update!(w,g,opt)
    end

    function Accuracy(w,data,lambda,binary)
        ncorrect = 0.0
        nsentences = 0.0
        J=0.0
        for batch in data
            for x in batch
                y,ytrue=predict(w,x)
                #ypred=convert(Array,y[:,end])
                ypred=convert(Array,y)
                ygold=ytrue
                if binary
                    ncorrect+= indmax(ypred)==1 && (ygold==1 || ygold==2)?1.0:0.0
                    ncorrect+= indmax(ypred)==2 && (ygold==1 || ygold==2)?1.0:0.0
                    ncorrect+= indmax(ypred)==4 && (ygold==4 || ygold==5)?1.0:0.0
                    ncorrect+= indmax(ypred)==5 && (ygold==4 || ygold==5)?1.0:0.0
                else
                    ncorrect+= indmax(ypred)==ygold?1.0:0.0
                end
            end
            J += loss(w,batch,lambda)
            nsentences+=length(batch)
        end
        tag_acc=ncorrect/nsentences
        tag_loss=J/length(data)

        return tag_acc,tag_loss
    end

    function main(args)
        t00=now();
        println(t00)
        s = ArgParseSettings()
        s.description = "Baseline LSTM"

        @add_arg_table s begin
            ("--usegpu"; action=:store_true; help="use GPU or not")
            ("--bilstm"; action=:store_true; help="use bidirectional or not")
            ("--binary"; action=:store_true; help="binary or fine-grained")
            ("--layer"; arg_type=Int; default=1; help="number of layers")
            ("--dropout"; arg_type=Float64; default=0.5; help="dropout ratio")
            ("--LR"; arg_type=Float64; default=0.05; help="learning rate")
            ("--lambda"; arg_type=Float64; default=0.0001; help="L2 regularization strength")
            ("--embed"; arg_type=Int; default=300; help="word embedding size")
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

        trn,dev,tst=load_treebank_data()
        l2i,w2i,i2l,i2w,w2g,words = build_treebank_vocabs(trn,dev,tst)
        println("number of words not in glove is ",length(words)-length(w2g))

        if o[:binary]
            trn=binarized(trn)
            dev=binarized(dev)
            tst=binarized(tst)
        end

        make_data!(trn,w2i,l2i,w2g,false)
        make_data!(dev,w2i,l2i,w2g,false)
        make_data!(tst,w2i,l2i,w2g,false)

        println(sum.(spans(trn[1],false)))

        #=if isfile("SST_seq.jld")
            trn=load("SST_seq.jld","trn")
            dev=load("SST_seq.jld","dev")
            tst=load("SST_seq.jld","tst")
        else
            trn=seq_data(trn)
            dev=seq_data(dev)
            tst=seq_data(tst)
            save("SST_seq.jld","trn",trn,"dev",dev,"tst",tst)
        end
        println(size(spans(trn[1][1],false)))=#



        trn=minibatch(trn,o[:batchsize])
        dev=minibatch(dev,o[:batchsize])
        tst=minibatch(tst,o[:batchsize])

        println(size(trn))


        nwords, ntags = length(w2i), length(l2i)
        w=optim=nothing
        w = initweights(atype,o[:bilstm],o[:layer],o[:dropout], ntags, o[:embed], o[:usegpu])
        optim=optimizers(w,Adagrad,lr=o[:LR])




        #trn = make_batches(trn, o[:batchsize])
        #dev = make_batches(dev, o[:batchsize])
        #tst = make_batches(tst, o[:batchsize])

        println("startup time: ", Int((now()-t00).value)*0.001); flush(STDOUT)

         for i in 1:o[:epochs]
             shuffle!(trn)
        @time for x in trn
                 train(w,x,o[:lambda],optim)
            end
            trnacc,trnloss = Accuracy(w, trn,o[:lambda],o[:binary])
            devacc,devloss = Accuracy(w, dev,o[:lambda],o[:binary])
            tstacc,tstloss = Accuracy(w, tst,o[:lambda],o[:binary])
            println("EPOCH: ",i)
            @printf("trnacc=%.4f,   trnloss=%f\n", trnacc,trnloss)
            @printf("devacc=%.4f,   devloss=%f\n", devacc,devloss)
            @printf("tstacc=%.4f,   tstloss=%f\n", tstacc,tstloss)
        end

    end

    splitdir(PROGRAM_FILE)[end] == "SST_lstm.jl" && main(ARGS)
end
