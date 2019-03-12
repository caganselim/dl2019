using Statistics
using Knet: Knet, dir, zeroone, progress, sgd, load, save, gc, Param, progress!
 KnetArray, gpu, Data, nll, relu, training, dropout, minibatch, param, param0
using Base.Iterators
using Plots; default(fmt=:png,ls=:auto)

include(Knet.dir("data", "cifar.jl"))

include("models.jl")


take_every(n,itr) = (x for (i,x) in enumerate(itr) if i % n == 1)
function train_results(file,model,from_scratch=true; o...)
    if (from_scratch)
        r = ((model(dtrn), model(dtst), zeroone(model,dtrn), zeroone(model,dtst))
             for x in take_every(length(dtrn), progress(sgd(model,repeat(dtrn,20)))))
        r = reshape(collect(Float32,flatten(r)),(4,:))
        Knet.save(file,"results",r)
        Knet.gc() # To save gpu memory
    else
        r = Knet.load(file,"results")
    end
    println(minimum(r,dims=2))
    return r
end

function load_data()
    @info("Loading CIFAR 10...")
    xtrn, ytrn, xtst, ytst, = cifar10()
    #= Subtract mean of each feature
    where each channel is considered as
    a single feature following the CNN
    convention=#
    mn = mean(xtrn, dims=(1,2,4))
    xtrn = xtrn .- mn
    xtst = xtst .- mn
    @info("Loaded CIFAR 10")
    return (xtrn, ytrn), (xtst, ytst)
end

(xtrn, ytrn), (xtst, ytst) = load_data()
@show size(xtrn)
@show size(ytrn)
@show size(xtst)
@show size(ytst)

# Need to reshape it to 2 dims for MLP
xtrn = reshape(xtrn, (32*32*3, :))
xtst = reshape(xtst, (32*32*3, :))

dtrn = minibatch(xtrn, ytrn, 50)
dtst = minibatch(xtst, ytst, 50)

model = Chain(Dense(32*32*3, 64), Dense(64, 32), Dense(32, 10, identity))

mlp = train_results("mlp.jld2", model, true)
plot([mlp[1,:], mlp[2,:]], labels=[:trnMLP :tstMLP], xlabel="Epochs", ylabel="Loss")
