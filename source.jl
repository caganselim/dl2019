using Statistics
using Knet: Knet, dir, zeroone, progress, sgd, load, save, gc, progress!, Param,
 KnetArray, gpu, Data, nll, relu, training, dropout, minibatch, param, param0,
 conv4, pool, mat, zeroone, sgd, adam, rmsprop, adagrad, sigm, softmax, tanh,
  batchnorm, bnparams, bnmoments, BNMoments, _update_moments!, _lazy_init!
  accuracy, xavier
using AutoGrad
using Base.Iterators
using Plots; default(fmt=:png,ls=:auto)
# using Profile
# using ProfileView


include(Knet.dir("data", "cifar.jl"))

include("models.jl")
include("wider.jl")
include("deeper.jl")

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
            new_r = reshape(collect(Float32,flatten(new_r)),(4,:))
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
    # (xtrn, ytrn), (xtst, ytst) = load_data()
    #
    # dtrn = minibatch(xtrn, ytrn, 50, xtype=atype())
    # dtst = minibatch(xtst, ytst, 50, xtype=atype())
    #
    # teacher = create_inception_bn_sm_narrow_model(3, 10)
    # results, teacher_trained = train_results(dtrn, dtst, "inception_sm_narrower.jld2", teacher, 3, false, false)
    # plot([results[1,:], results[2,:]], labels=["trn teacher" "tst teacher"], xlabel="Epochs", ylabel="Loss")
    # println("Teacher results: ", results)
    #
    # growth_ratio = 1.0/sqrt(0.3)

    # wider = deepcopy(teacher)
    # wider_inceptionA(wider.layers[3], wider.layers[4], growth_ratio)
    # wider_inceptionA(wider.layers[4], wider.layers[5], wider.layers[7], growth_ratio)
    # wider_inceptionB(wider.layers[5], wider.layers[7], growth_ratio)
    #
    # padded = deepcopy(teacher)
    # random_pad_inceptionA(padded.layers[3], padded.layers[4], growth_ratio)
    # random_pad_inceptionA(padded.layers[4], padded.layers[5], padded.layers[7], growth_ratio)
    # random_pad_inceptionB(padded.layers[5], padded.layers[7], growth_ratio)
    #
    # results, wider_trained = train_results(dtrn, dtst, "inception_sm_wider2.jld2", wider, 5, true)
    # plot([results[1,:], results[2,:]], labels=["trn wider" "tst wider"], xlabel="Epochs", ylabel="Loss")
    # println("Wider results: ", results)
    #
    # results, padded_trained = train_results(dtrn, dtst, "inception_sm_padded2.jld2", padded, 5, true)
    # plot([results[1,:], results[2,:]], labels=["trn padded" "tst padded"], xlabel="Epochs", ylabel="Loss")
    # println("Padded results: ", results)

    m = create_cnn_model(3, 10, false)
    deeper_conv(m.layers[4])
end

Knet.gc()
test_deeper_conv()
# test_random_pad_inception()
# test_wider_inception()
# test_wider_conv()
# test_wider_mlp()
