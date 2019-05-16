using Statistics
using Knet: Knet, dir, zeroone, progress, sgd, load, save, gc, progress!, Param,
 KnetArray, gpu, Data, nll, relu, training, dropout, minibatch, param, param0,
 conv4, pool, mat, zeroone, sgd, adam, rmsprop, adagrad, sigm, softmax, tanh,
  batchnorm, bnparams, bnmoments, BNMoments, _update_moments!, _lazy_init!,
  accuracy, xavier
using AutoGrad
using Base.Iterators
using DelimitedFiles


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
take_every(n,itr) = (x for (i,x) in enumerate(itr) if i % n == 0)

"""
Trains a model, tests it every epoch on training and testing data.
Saves results to a file and can load them back. Returns the results.
"""
function train_results(dtrn, dtst, file, model, epochs=100, from_scratch=true, cont_from_save=false; o...)
    file = string("saved/", file)
    if !from_scratch
        try
            r, model = Knet.load(file, "results", "model")
            if cont_from_save
                new_r = ((model(dtrn), model(dtst), accuracy(model, dtrn), accuracy(model, dtst))
                     for x in take_every(length(dtrn), progress(adam(model, repeat(dtrn,epochs)))))
                new_r = reshape(collect(Float32,flatten(new_r)),(4,:))
                r = hcat(r, new_r)
                Knet.save(file, "results", r, "model", model)
                Knet.gc() # To save gpu memory
            end
        catch SystemError
            println("File not found. Running from scratch.")
            from_scratch = true
        end
    end

    if from_scratch
        r_first = [model(dtrn), model(dtst), accuracy(model, dtrn), accuracy(model, dtst)]
        r = ((model(dtrn), model(dtst), accuracy(model, dtrn), accuracy(model, dtst))
             for x in take_every(length(dtrn), progress(adam(model, repeat(dtrn,epochs)))))
        r = reshape(collect(Float32,flatten(r)),(4,:))
        r = hcat(r_first, r)
        Knet.save(file, "results", r, "model", model)
        Knet.gc() # To save gpu memory
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
Net2WiderNet experiment
-----------------------
Each Inception module in a smaller Inception network is widened by a factor of
sqrt(0.3) using Net2WiderNet and the baseline method random padding.
"""
function wider_experiment(dtrn, dtst)
    teacher = create_inception_bn_sm_model(3, 10)
    results, teacher = train_results(dtrn, dtst, "inception_sm.jld2", teacher, 5, false)
    println("Teacher results: ", results)
    writedlm("res/wider_res_teacher.txt", results)

    growth_ratio = 1.0/sqrt(0.3)
    noise = 0.03

    wider = deepcopy(teacher)
    wider_inceptionA(wider.layers[3], wider.layers[4], growth_ratio, noise)
    wider_inceptionA(wider.layers[4], wider.layers[5], wider.layers[7], growth_ratio, noise)
    wider_inceptionB(wider.layers[5], wider.layers[7], growth_ratio, noise)

    padded = teacher
    random_pad_inceptionA(padded.layers[3], padded.layers[4], growth_ratio)
    random_pad_inceptionA(padded.layers[4], padded.layers[5], padded.layers[7], growth_ratio)
    random_pad_inceptionB(padded.layers[5], padded.layers[7], growth_ratio)

    Knet.gc()

    results, wider = train_results(dtrn, dtst, "inception_sm_wider.jld2", wider, 5, true)
    println("Wider results: ", results)
    writedlm("res/wider_res_wider.txt", results)

    results, padded = train_results(dtrn, dtst, "inception_sm_padded.jld2", padded, 5, true)
    println("Padded results: ", results)
    writedlm("res/wider_res_padded.txt", results)
end

"""
Net2DeeperNet experiment
------------------------
Each Inception module in a smaller Inception network is deepened by 2 layers
using Net2DeeperNet and the baseline method random initialization.
"""
function deeper_experiment(dtrn, dtst)
    teacher = create_inception_bn_sm_model(3, 10)
    results, teacher = train_results(dtrn, dtst, "inception_sm.jld2", teacher, 5, false)
    println("Teacher results: ", results)
    writedlm("res/deeper_res_teacher.txt", results)

    deeper = deepcopy(teacher)
    deeper_inception(deeper.layers, 3, dtrn)
    deeper_inception(deeper.layers, 4, dtrn)
    deeper_inception(deeper.layers, 5, dtrn)

    rand_deeper = create_inception_bn_sm_model(3, 10, true)

    results, deeper = train_results(dtrn, dtst, "inception_sm_deeper.jld2", deeper, 5, true)
    println("Deeper results: ", results)
    writedlm("res/deeper_res_deeper.txt", results)

    results, rand = train_results(dtrn, dtst, "inception_sm_rand_deeper.jld2", rand_deeper, 5, true)
    println("Rand Deeper results: ", results)
    writedlm("res/deeper_res_rand.txt", results)
end

"""
Exploring design space experiment
---------------------------------
The design space is explored by using Net2WiderNet with a factor of sqrt(2) and
Net2DeeperNet with 4 layers per module layer. An additional student that is both
wider and deeper is also explored.
"""
function explore_experiment(dtrn, dtst)
    teacher = create_inception_bn_sm_model(3, 10)
    results, teacher = train_results(dtrn, dtst, "inception_sm.jld2", teacher, 5, false)
    println("Teacher results: ", results)
    writedlm("res/explore_res_teacher.txt", results)

    widening_factor = sqrt(2.0)
    wider = deepcopy(teacher)
    wider_inceptionA(wider.layers[3], wider.layers[4], widening_factor)
    wider_inceptionA(wider.layers[4], wider.layers[5], wider.layers[7], widening_factor)
    wider_inceptionB(wider.layers[5], wider.layers[7], widening_factor)

    deepening_factor = 4
    deeper = deepcopy(teacher)
    deeper_inception(deeper.layers, 3, dtrn, deepening_factor)
    deeper_inception(deeper.layers, 4, dtrn, deepening_factor)
    deeper_inception(deeper.layers, 5, dtrn, deepening_factor)

    results, wider = train_results(dtrn, dtst, "inception_sm_exp_wider.jld2", wider, 5, true)
    println("Wider results: ", results)
    writedlm("res/explore_res_wider.txt", results)

    results, deeper = train_results(dtrn, dtst, "inception_sm_exp_deeper.jld2", deeper, 5, true)
    println("Deeper results: ", results)
    writedlm("res/explore_res_deeper.txt", results)

    # New one with both widening and deepening, but with smaller factors
    bigger = deepcopy(teacher)
    widening_factor = 1/sqrt(0.3)
    wider_inceptionA(bigger.layers[3], bigger.layers[4], widening_factor)
    wider_inceptionA(bigger.layers[4], bigger.layers[5], bigger.layers[7], widening_factor)
    wider_inceptionB(bigger.layers[5], bigger.layers[7], widening_factor)
    deeper_inception(bigger.layers, 3, dtrn, 2)
    deeper_inception(bigger.layers, 4, dtrn, 2)
    deeper_inception(bigger.layers, 5, dtrn, 2)

    results, bigger = train_results(dtrn, dtst, "inception_sm_exp_bigger.jld2", bigger, 5, true)
    println("Bigger results: ", results)
    writedlm("res/explore_res_bigger.txt", results)
end

"""
Added Noise experiment
----------------------
Experiment to find out how much noise one should add after Net2WiderNet. Results
show that Gaussian noise with 0.05 gives good results.
"""
function noise_experiment(dtrn, dtst)
    teacher = create_inception_bn_sm_model(3, 10)
    results, teacher = train_results(dtrn, dtst, "inception_sm.jld2", teacher, 5, false)

    growth_ratio = 1.0/sqrt(0.3)

    wider_no_noise = deepcopy(teacher)
    wider_inceptionA(wider_no_noise.layers[3], wider_no_noise.layers[4], growth_ratio, 0)
    wider_inceptionA(wider_no_noise.layers[4], wider_no_noise.layers[5], wider_no_noise.layers[7], growth_ratio, 0)
    wider_inceptionB(wider_no_noise.layers[5], wider_no_noise.layers[7], growth_ratio, 0)

    wider_noise_1 = deepcopy(teacher)
    wider_inceptionA(wider_noise_1.layers[3], wider_noise_1.layers[4], growth_ratio, 0.01)
    wider_inceptionA(wider_noise_1.layers[4], wider_noise_1.layers[5], wider_noise_1.layers[7], growth_ratio, 0.01)
    wider_inceptionB(wider_noise_1.layers[5], wider_noise_1.layers[7], growth_ratio, 0.01)

    wider_noise_2 = deepcopy(teacher)
    wider_inceptionA(wider_noise_2.layers[3], wider_noise_2.layers[4], growth_ratio, 0.05)
    wider_inceptionA(wider_noise_2.layers[4], wider_noise_2.layers[5], wider_noise_2.layers[7], growth_ratio, 0.05)
    wider_inceptionB(wider_noise_2.layers[5], wider_noise_2.layers[7], growth_ratio, 0.05)

    wider_noise_3 = deepcopy(teacher)
    wider_inceptionA(wider_noise_3.layers[3], wider_noise_3.layers[4], growth_ratio, 0.1)
    wider_inceptionA(wider_noise_3.layers[4], wider_noise_3.layers[5], wider_noise_3.layers[7], growth_ratio, 0.1)
    wider_inceptionB(wider_noise_3.layers[5], wider_noise_3.layers[7], growth_ratio, 0.1)


    results, wider = train_results(dtrn, dtst, "inception_sm_exp_wider_no_noise.jld2", wider_no_noise, 5, true)
    println("Wider no noise results: ", results)
    writedlm("res/noise_res_no.txt", results)

    results, wider = train_results(dtrn, dtst, "inception_sm_exp_wider_noise_2.jld2", wider_noise_2, 5, true)
    println("Wider noise 2 results: ", results)
    writedlm("res/noise_res_2.txt", results)

    results, wider = train_results(dtrn, dtst, "inception_sm_exp_wider_noise_3.jld2", wider_noise_3, 5, true)
    println("Wider noise 3 results: ", results)
    writedlm("res/noise_res_3.txt", results)
end

function main()
    (xtrn, ytrn), (xtst, ytst) = load_data()

    dtrn = minibatch(xtrn, ytrn, 50, xtype=atype())
    dtst = minibatch(xtst, ytst, 50, xtype=atype())

    wider_experiment(dtrn, dtst)
    deeper_experiment(dtrn, dtst)
    explore_experiment(dtrn, dtst)
    noise_experiment(dtrn, dtst)
end

function perform_tests()
    test_wider_mlp()
    # test_wider_conv()
    # test_wider_inception()
    # test_random_pad_inception()
    # test_deeper_conv()
    # test_deeper_inception()
end

# main()
perform_tests()
