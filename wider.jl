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
Grows one of the layers of the model to a new one with more hidden units.
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

"""
Net2WiderNet method for Convs.
Grows one of the layers of the model to a new one with more filter channels.
Function is preserved during the widening.
"""
function wider_conv(layer, next_layer, new_channel_count, add_noise=true)
    old_channel_count = size(layer.w, 4)
    if old_channel_count >= new_channel_count
        println("New channel count must be greater than old channel count")
        return nothing
    end

    # new_model = deepcopy(model)

    # Initialize number of extra channels, extra arrays, random mapping, number of copies
    extra_count = new_channel_count - old_channel_count
    extra_w = Array{Float32}(undef, (size(layer.w)[1:3]..., extra_count))
    extra_next_w = Array{Float32}(undef, (size(next_layer.w)[1:2]..., extra_count, size(next_layer.w)[4]))
    extra_b = Array{Float32}(undef, (1, 1, extra_count, 1))

    if layer.bn_params != nothing
        extra_bn_scales = atype()(undef, (extra_count,))
        extra_bn_biases = atype()(undef, (extra_count,))
    end

    extra_mapping = rand(1:old_channel_count, extra_count)
    copy_counts = ones(old_channel_count)
    for i in 1:extra_count
        copy_counts[extra_mapping[i]] += 1
    end
    # Divide weights that are copied more than once
    next_layer.w ./= atype()(reshape(copy_counts, (1,1,:,1)))

    # Set the extra weights
    for i in 1:extra_count
        extra_w[:, :, :, i] = convert(Array{Float32}, layer.w)[:, :, :, extra_mapping[i]]
        extra_next_w[:, :, i, :] = convert(Array{Float32}, next_layer.w)[:, :, extra_mapping[i], :]
        extra_b[1, 1, i, 1] = layer.b[1, 1, extra_mapping[i], 1]
        if layer.bn_params != nothing
            extra_bn_scales[i] = layer.bn_params[extra_mapping[i]]
            extra_bn_biases[i] = layer.bn_params[old_channel_count+extra_mapping[i]]
        end
    end

    extra_w = atype()(extra_w)
    extra_next_w = atype()(extra_next_w)
    extra_b = atype()(extra_b)

    # Update the weights
    old_size = size(layer.w)
    w_old_2d = reshape(layer.w, :, size(layer.w)[end])
    w_extra_2d = reshape(extra_w, :, size(extra_w)[end])
    w_new_2d = hcat(w_old_2d, w_extra_2d)
    layer.w = Param(reshape(w_new_2d, old_size[1], old_size[2], old_size[3], :))

    layer.b = Param(reshape(vcat(reshape(layer.b, (size(layer.b, 3),)), reshape(extra_b, (extra_count,))), (1, 1, new_channel_count, 1)))

    old_size = size(next_layer.w)
    w_old_2d = reshape(next_layer.w, :, size(next_layer.w)[end])
    w_extra_2d = reshape(extra_next_w, :, size(extra_next_w)[end])
    w_new_2d = vcat(w_old_2d, w_extra_2d)
    next_layer.w = Param(reshape(w_new_2d, old_size[1], old_size[2], :, old_size[4]))
    if layer.bn_params != nothing
        layer.bn_params = Param(cat(layer.bn_params[1:old_channel_count], extra_bn_scales, layer.bn_params[old_channel_count+1:end], extra_bn_biases, dims=1))
    end
end

function test_wider_conv()
    (xtrn, ytrn), (xtst, ytst) = load_data()

    dtrn = minibatch(xtrn, ytrn, 50, xtype=atype())
    dtst = minibatch(xtst, ytst, 50, xtype=atype())

    narrow = 64
    wide = 72
    chosen_layer = 4

    cnn_model = create_cnn_bn_model(3, 10)
    cnn_results, cnn_model = train_results(dtrn, dtst, "cnn.jld2", cnn_model, 5, false)
    cnn_wider = deepcopy(cnn_model)
    wider_conv(cnn_wider.layers[chosen_layer], cnn_wider.layers[chosen_layer+1], wide)

    @assert (size(cnn_model.layers[chosen_layer].w, 4) == narrow &&
            size(cnn_model.layers[chosen_layer].b, 3) == narrow &&
            size(cnn_model.layers[chosen_layer+1].w, 3) == narrow) "Old model is modified"

    @assert (size(cnn_wider.layers[chosen_layer].w, 4) == wide &&
            size(cnn_wider.layers[chosen_layer].b, 3) == wide &&
            size(cnn_wider.layers[chosen_layer+1].w, 3) == wide) "New model is not widened correctly"

    sames = 0
    total = 0
    for (x, y) in dtst
        y_olds = cnn_model(x)
        y_news = cnn_wider(x)

        sames += sum(y_olds .- y_news .< 0.01)
        total += length(y_olds)
    end

    @assert sames/total == 1 "Function is not preserved"
    println("wider conv test passed")
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
    println("wider mlp test passed")
end
