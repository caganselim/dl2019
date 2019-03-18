using Statistics
using Knet: Knet, dir, zeroone, progress, sgd, load, save, gc, progress!, Param,
 KnetArray, gpu, Data, nll, relu, training, dropout, minibatch, param, param0,
 accuracy
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
        r = ((model(dtrn), model(dtst), accuracy(model,dtrn), accuracy(model,dtst))
             for x in take_every(length(dtrn), progress(sgd(model,repeat(dtrn,epochs)))))
        r = reshape(collect(Float32,flatten(r)),(4,:))
        Knet.save(file, "results", r, "model", clean(model))
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
    atype() = (at == nothing) ? (at = (gpu() >= 0 ? KnetArray : Array)) : at
end

(xtrn, ytrn), (xtst, ytst) = load_data()
@show size(xtrn)
@show size(ytrn)
@show size(xtst)
@show size(ytst)

# Need to reshape it to 2 dims for MLP
xtrn = reshape(xtrn, (32*32*3, :))
xtst = reshape(xtst, (32*32*3, :))

dtrn = minibatch(xtrn, ytrn, 50, xtype=atype())
dtst = minibatch(xtst, ytst, 50, xtype=atype())

mlp_model = create_mlp_model(32*32*3, 10, 16, 16)

mlp_results, mlp_model = train_results("mlp.jld2", mlp_model, 5, false)
plot([mlp_results[1,:], mlp_results[2,:]], labels=[:trnMLP :tstMLP], xlabel="Epochs", ylabel="Loss")
plot([mlp_results[3,:], mlp_results[4,:]], labels=[:trnMLP :tstMLP], xlabel="Epochs", ylabel="Accuracy")

# MLP wider is randomly padded
mlp_wider = random_pad_mlp(mlp_model, 1, 32)
# MLP deeper is completely randomly initialized
mlp_deeper = create_mlp_model(32*32*3, 10, 16, 16, 16)

println("MLP train loss: ", mlp_model(dtrn))
println("MLP test loss: ", mlp_model(dtst))
println("MLP wider train loss: ", mlp_wider(dtrn))
println("MLP wider test loss: ", mlp_wider(dtst))
println("MLP deeper train loss: ", mlp_deeper(dtrn))
println("MLP deeper test loss: ", mlp_deeper(dtst))

println("MLP train accuracy: ", accuracy(mlp_model,dtrn))
println("MLP test accuracy: ", accuracy(mlp_model,dtst))
println("MLP wider train accuracy: ", accuracy(mlp_wider,dtrn))
println("MLP wider test accuracy: ", accuracy(mlp_wider,dtst))
println("MLP deeper train accuracy: ", accuracy(mlp_deeper,dtrn))
println("MLP deeper test accuracy: ", accuracy(mlp_deeper,dtst))

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
