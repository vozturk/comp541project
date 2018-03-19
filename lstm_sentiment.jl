
# generic lstm baseline model

# USAGE
# Change args in the Main() function according to the model

function helper(t,ind=Any[])
    s= nothing
        if length(t.children) == 1 && isleaf(t.children[1])
            l = t.children[1]
            s=l.data
        elseif length(t.children) == 2
            t1,t2 = t.children[1], t.children[2]
            helper(t1,ind)
            helper(t2,ind)
        else
            error("invalid tree")
        end
        s!=nothing ? push!(ind,s): ind
        return ind
    end

    function seq_data(data)
        samples=Any[]
        for i in 1:length(data)
            push!(samples,helper(data[i]))
        end
        return samples
    end


    function make_batches(data, batchsize)
        batches = []
        sorted = sort(data, by=length, rev=true)
        for k = 1:batchsize:length(sorted)
            lo = k
            hi = min(k+batchsize-1, length(sorted))
            samples = sorted[lo:hi]
            batch = make_batch(samples)
            push!(batches, batch)
        end
        return batches
    end

    function make_batch(samples)
        input = Int[]
        output = Int[]
        longest = length(samples[1])
        batchsizes = zeros(Int, longest)
        s=length(samples)
        for i in 1:longest
            for j in 1:s
                if length(samples[j])>=i
                    push!(input, trnseq[j][i] )
                    batchsizes[i] +=1
                end
                i==1? push!(output,trn[j].data):output
            end
        end
        return input, output, batchsizes
    end

    function initweights(atype,dir,lay,pdrop,hidden,words, tags, embed, usegpu, winit=0.01)
        w = Array{Any}(5)
        we(d...)=atype(winit.*randn(d...))
        bi(d...)=atype(zeros(d...))
        w[2]=we(tags,dir*hidden)
        w[3]=bi(tags,1)
        w[4]=we(embed,words)
        w[5],w[1]=rnninit(embed,hidden;bidirectional=dir,rnnType=:lstm,numLayers=lay,dropout=pdrop)
        return w
    end

    function predict(ws, xs, batchsizes)
        # YOUR ANSWER
         x = ws[4][:,xs]
        (y,_) = rnnforw(ws[5],ws[1],x,batchSizes=batchsizes)
        y=ws[2]*y .+ ws[3]
        return y
    end

    # our loss function
    function loss(w, x, ygold, batchsizes,lambda)
        # YOUR ANSWER
        for i in 1:length(w)-1
            sum_par+=sum(w[i].^2)    
        end
         l=nll(predict(w,x,batchsizes),ygold) + (lambda/2).sum_par
         return l
    end

    function Accuracy(w, batches,lambda)
        # YOUR ANSWER
       ncorrect = 0.0
        nsentence = 0.0
        for (x,y,z) in batches
            ypred=predict(w,x,z)
            loss += loss(w,x,y,z,lambda)
            yped=ypred[:,end]
            yhat=convert(Array,ypred)
            for i in 1:length(y)
                ncorrect+= indmax(yhat[:,i])==y[i]?1.0:0.0
            end
            nsentence += length(y)
        end
        tag_acc=ncorrect/nsentence
        tag_loss=loss/nsentence

        return tag_acc,tag_loss
    end

    function main(args)
        s = ArgParseSettings()
        s.description = "Baseline LSTM"

        @add_arg_table s begin
            ("--usegpu"; action=:store_true; help="use GPU or not")
            ("--bidirectional"; arg_type=Bool; default=false; help="use bidirectional or not")
            ("--layer"; arg_type=Int; default=1; help="number of layers")
            ("--dropout"; arg_type=Float32; default=0.5; help="dropout ratio")
            ("--lambda"; arg_type=Float32; default=0.0001; help="L2 regularization strength")
            ("--embed"; arg_type=Int; default=300; help="word embedding size")
            ("--hidden"; arg_type=Int; default=168; help="LSTM hidden size")
            ("--epochs"; arg_type=Int; default=20; help="number of training epochs")
            ("--report"; arg_type=Int; default=500; help="report period in iters")
            ("--valid"; arg_type=Int; default=10000; help="valid period in iters")
            ("--seed"; arg_type=Int; default=-1; help="random seed")
            ("--batchsize"; arg_type=Int; default=25; help="batchsize")
        end

        isa(args, AbstractString) && (args=split(args))
        o = parse_args(args, s; as_symbols=true)
        o[:seed] > 0 && Knet.setseed(o[:seed])
        atype = o[:atype] = !o[:usegpu] ? Array{Float32} : KnetArray{Float32}

        include(Pkg.dir("Knet/data/treebank.jl"))
        UNK="_UNK_"
        trn,dev = load_treebank_data("train")
        l2i, w2i, i2l, i2w = build_treebank_vocabs(trn)
        make_data!(trn,w2i,l2i)
        make_data!(dev,w2i,l2i)
        trnseq= seq_data(trn)
        devseq=seq_data(dev)



        # build model
        nwords, ntags = length(w2i), length(l2i)
        w = initweights(
            atype, o[:hidden],o[:bidirectional],o[:layer],o[:dropout], nwords, ntags, o[:embed], o[:usegpu])

        # make batches
        trn = make_batches(data.trn, o[:batchsize])
        dev = make_batches(data.dev, o[:batchsize])


            # validation
        trnacc,trnloss = Accuracy(w, dev,o[:lambda])
        devacc,devloss = Accuracy(w, dev,o[:lambda])

            # report
        @printf("trnacc=%.4f,trnloss=%f\n", trnacc,trnloss)
        @printf("devacc=%.4f,devloss=%f\n", devacc,devloss)

    end

    main("--usegpu")
