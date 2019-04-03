using Statistics
using Knet: Knet, dir, zeroone, progress, sgd, load, save, gc, progress!, Param,
 KnetArray, gpu, Data, nll, relu, training, dropout, minibatch, param, param0,
 conv4, pool, mat, zeroone, sgd, adam, rmsprop, adagrad, sigm, softmax, tanh,
 batchnorm, bnparams, bnmoments
using AutoGrad
using Base.Iterators
using Plots; default(fmt=:png,ls=:auto)

include(Knet.dir("data", "cifar.jl"))

include("models.jl")

"Take every nth element in an iterator"
take_every(n,itr) = (x for (i,x) in enumerate(itr) if i % n == 1)

"""
Trains a model, tests it every epoch on training and testing data.
Saves results to a file and can load them back. Returns the results.
"""
function train_results(file,model,epochs=100,from_scratch=true; o...)
    if (from_scratch)
        r = ((model(dtrn), model(dtst), zeroone(model,dtrn), zeroone(model,dtst))
             for x in take_every(length(dtrn), progress(sgd(model,repeat(dtrn,epochs)))))
        r = reshape(collect(Float32,flatten(r)),(4,:))
        Knet.save(file, "results", r, "model", model)
        Knet.gc() # To save gpu memory
    else
        r, model = Knet.load(file, "results", "model")
    end
    println(minimum(r,dims=2))
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

"""
Baseline method for Net2WiderNet.
Grows one of the layers of the model to a new one with a larger size.
The old units are copied as is, new units are initialized randomly.
"""
function random_pad_mlp(model, changing_layer, new_unit_count)
    old_unit_count = size(model.layers[changing_layer].w, 1)
    if old_unit_count >= new_unit_count
        println("New unit count must be greater than old unit count")
        return nothing
    end

    padding_count = new_unit_count - old_unit_count

    padding_1 = param(padding_count, size(model.layers[changing_layer].w, 2))
    padding_2 = param(size(model.layers[changing_layer+1].w, 1), padding_count)
    padding_b = param0(padding_count)

    new_model = deepcopy(model)
    new_model.layers[changing_layer].w = Param(vcat(new_model.layers[changing_layer].w, padding_1))
    new_model.layers[changing_layer].b = Param(vcat(new_model.layers[changing_layer].b, padding_b))
    new_model.layers[changing_layer+1].w = Param(hcat(new_model.layers[changing_layer+1].w, padding_2))

    return new_model
end

# The global device setting (to reduce gpu() calls)
let at = nothing
    global atype
    atype() = (at == nothing) ? (at = (gpu() >= 0 ? KnetArray{Float32} : Array{Float32})) : at
end

(xtrn, ytrn), (xtst, ytst) = load_data()
@show size(xtrn)
@show size(ytrn)
@show size(xtst)
@show size(ytst)

# Need to reshape it to 2 dims for MLP
# xtrn = reshape(xtrn, (32*32*3, :))
# xtst = reshape(xtst, (32*32*3, :))

dtrn = minibatch(xtrn, ytrn, 20, xtype=atype())
dtst = minibatch(xtst, ytst, 25, xtype=atype())
# mlp_model = create_mlp_model(32*32*3, 10, 16, 16)
#
# mlp_results, mlp_model = train_results("mlp.jld2", mlp_model, 5, false)
# plot([mlp_results[1,:], mlp_results[2,:]], labels=[:trnMLP :tstMLP], xlabel="Epochs", ylabel="Loss")
#
# mlp_wider = random_pad_mlp(mlp_model, 1, 20)
#
# @show mlp_model(dtrn)
# @show mlp_model(dtrn)
# @show mlp_wider(dtrn)
# @show mlp_wider(dtst)

inception_cifar = create_inception_bn_smaller_model(3, 10)
inc_results, inc_model = train_results("inception_smaller.jld2", inception_cifar, 3, true)
plot([inc_results[1,:], inc_results[2,:]], labels=[:trnINC :tstINC], xlabel="Epochs", ylabel="Loss")

# mlp1 = create_mlp_model(2, 3, 2)
# mlp2 = random_pad_mlp(mlp1, 1, 3)
#
# println("MLP1")
# for l in mlp1.layers
#     @show l
# end
# println("MLP2")
# for l in mlp2.layers
#     @show l
# end
