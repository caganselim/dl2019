using Statistics
using Knet: Knet, dir, zeroone, progress, sgd, load, save, gc, progress!, Param,
 KnetArray, gpu, Data, nll, relu, training, dropout, minibatch, param, param0,
 conv4, pool, mat, zeroone, sgd, adam, rmsprop, adagrad, sigm, softmax, tanh,
 batchnorm, bnparams, bnmoments, accuracy
using AutoGrad
using Base.Iterators
using Plots; default(fmt=:png,ls=:auto)
# using Profile
# using ProfileView


include(Knet.dir("data", "cifar.jl"))

include("models.jl")
include("wider.jl")

# The global device setting (to reduce gpu() calls)
let at = nothing
    global atype
    atype() = (at == nothing) ? (at = (gpu() >= 0 ? KnetArray{Float32} : Array{Float32})) : at
end

"Take every nth element in an iterator"
take_every(n,itr) = (x for (i,x) in enumerate(itr) if i % n == 1)

"""
Trains a model, tests it every epoch on training and testing data.
Saves results to a file and can load them back. Returns the results.
"""
function train_results(dtrn, dtst, file, model, epochs=100, from_scratch=true, cont_from_save=false; o...)
    if from_scratch
        r = ((model(dtrn), model(dtst), accuracy(model, dtrn), accuracy(model, dtst))
             for x in take_every(length(dtrn), progress(adam(model, repeat(dtrn,epochs)))))
        r = reshape(collect(Float32,flatten(r)),(4,:))
        Knet.save(file, "results", r, "model", model)
        Knet.gc() # To save gpu memory
    else
        r, model = Knet.load(file, "results", "model")
        if cont_from_save
            # todo: continue training doesn't work for some reason
            new_r = ((model(dtrn), model(dtst), accuracy(model, dtrn), accuracy(model, dtst))
                 for x in take_every(length(dtrn), progress(adam(model, repeat(dtrn,epochs)))))
            new_r = reshape(collect(Float32,flatten(r)),(4,:))
            r = hcat(r, new_r)
            Knet.save(file, "results", r, "model", model)
            Knet.gc() # To save gpu memory
        end
    end
    println("Trn/Tst loss: ", minimum(r[1:2, :], dims=2),
            "Trn/Tst acc: ", maximum(r[3:4, :], dims=2))
    return r, model
end

"Loads the CIFAR-10 dataset"
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

function main()
    (xtrn, ytrn), (xtst, ytst) = load_data()

    dtrn = minibatch(xtrn, ytrn, 50, xtype=atype())
    dtst = minibatch(xtst, ytst, 50, xtype=atype())

    inception_cifar = create_inception_bn_smaller_model(3, 10)
    inc_results, inc_model = train_results(dtrn, dtst, "inception_smaller.jld2", inception_cifar, 50, true)
    plot([inc_results[1,:], inc_results[2,:]], labels=[:trnINC :tstINC], xlabel="Epochs", ylabel="Loss")
end

# main()
test_wider_conv()
# test_wider_mlp()
