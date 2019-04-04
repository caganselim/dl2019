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

"""
Net2WiderNet method for MLPs.
Grows one of the layers of the model to a new one with a larger size.
Function is preserved during the widening.
"""
function wider_mlp(model, changing_layer, new_unit_count, add_noise=true)
    old_unit_count = size(model.layers[changing_layer].w, 1)
    if old_unit_count >= new_unit_count
        println("New unit count must be greater than old unit count")
        return nothing
    end

    new_model = deepcopy(model)
    layer = new_model.layers[changing_layer]
    next_layer = new_model.layers[changing_layer + 1]

    # Initialize number of extra units, extra arrays, random mapping, number of copies
    extra_count = new_unit_count - old_unit_count
    extra_w = atype()(undef, (extra_count, size(layer.w, 2)))
    extra_next_w = atype()(undef, (size(next_layer.w, 1), extra_count))
    extra_b = atype()(undef, (extra_count))

    extra_mapping = rand(1:old_unit_count, extra_count)
    copy_counts = ones(old_unit_count)
    for i in 1:extra_count
        copy_counts[extra_mapping[i]] += 1
    end
    # Divide weights that are copied more than once
    next_layer.w ./= atype()(copy_counts')

    # Set the extra weights
    for i in 1:extra_count
        extra_w[i, :] = layer.w[extra_mapping[i], :]
        extra_next_w[:, i] = next_layer.w[:, extra_mapping[i]]
        extra_b[i] = layer.b[extra_mapping[i]]
    end

    # Update the weights
    layer.w = Param(vcat(layer.w, extra_w))
    layer.b = Param(vcat(layer.b, extra_b))
    next_layer.w = Param(hcat(next_layer.w, extra_next_w))
    return new_model
end

function test_wider_mlp()
    (xtrn, ytrn), (xtst, ytst) = load_data()

    # Need to reshape it to 2 dims for MLP
    xtrn = reshape(xtrn, (32*32*3, :))
    xtst = reshape(xtst, (32*32*3, :))

    dtrn = minibatch(xtrn, ytrn, 50, xtype=atype())
    dtst = minibatch(xtst, ytst, 50, xtype=atype())

    narrow = 16
    wide = 20
    chosen_layer = 1

    mlp_model = create_mlp_model(32*32*3, 10, narrow, 16)
    mlp_results, mlp_model = train_results(dtrn, dtst, "mlp.jld2", mlp_model, 100, true)
    plot([mlp_results[1,:], mlp_results[2,:]], labels=[:trnMLP :tstMLP], xlabel="Epochs", ylabel="Loss")
    mlp_wider = wider_mlp(mlp_model, chosen_layer, wide)

    @assert (size(mlp_model.layers[chosen_layer].w, 1) == narrow &&
            size(mlp_model.layers[chosen_layer].b, 1) == narrow &&
            size(mlp_model.layers[chosen_layer+1].w, 2) == narrow) "Old model is modified"

    @assert (size(mlp_wider.layers[chosen_layer].w, 1) == wide &&
            size(mlp_wider.layers[chosen_layer].b, 1) == wide &&
            size(mlp_wider.layers[chosen_layer+1].w, 2) == wide) "New model is not widened correctly"

    sames = 0
    total = 0
    for (x, y) in dtst
        y_olds = mlp_model(x)
        y_news = mlp_wider(x)

        sames += sum(y_olds .- y_news .< 0.01)
        total += length(y_olds)
    end
    @assert sames/total == 1 "Function is not preserved"

end

function main()
    (xtrn, ytrn), (xtst, ytst) = load_data()

    dtrn = minibatch(xtrn, ytrn, 50, xtype=atype())
    dtst = minibatch(xtst, ytst, 50, xtype=atype())

    inception_cifar = create_inception_bn_smaller_model(3, 10)
    inc_results, inc_model = train_results(dtrn, dtst, "inception_smaller.jld2", inception_cifar, 4, true)
    plot([inc_results[1,:], inc_results[2,:]], labels=[:trnINC :tstINC], xlabel="Epochs", ylabel="Loss")
end

main()
# test_wider_mlp()
