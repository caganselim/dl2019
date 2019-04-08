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

    # Initialize number of extra channels, extra arrays, random mapping, number of copies
    extra_count = new_channel_count - old_channel_count
    extra_w = Array{Float32}(undef, (size(layer.w)[1:3]..., extra_count))
    extra_next_w = next_layer != nothing ? Array{Float32}(undef, (size(next_layer.w)[1:2]..., extra_count, size(next_layer.w)[4])) : nothing
    extra_b = Array{Float32}(undef, (1, 1, extra_count, 1))

    if layer.bn_params != nothing
        extra_bn_scales = atype()(undef, (extra_count,))
        extra_bn_biases = atype()(undef, (extra_count,))
    end

    extra_mapping = rand(1:old_channel_count, extra_count)

    if next_layer != nothing
        copy_counts = ones(old_channel_count)
        for i in 1:extra_count
            copy_counts[extra_mapping[i]] += 1
        end
        # Divide weights that are copied more than once
        next_layer.w ./= atype()(reshape(copy_counts, (1,1,:,1)))
    end

    array_layer_w = convert(Array{Float32}, layer.w)
    array_next_layer_w = next_layer != nothing ? convert(Array{Float32}, next_layer.w) : nothing
    # Set the extra weights
    for i in 1:extra_count
        extra_w[:, :, :, i] = array_layer_w[:, :, :, extra_mapping[i]] .+ (add_noise ? randn()*0.01 : 0)
        if next_layer != nothing
            extra_next_w[:, :, i, :] = array_next_layer_w[:, :, extra_mapping[i], :] .+ (add_noise ? randn()*0.01 : 0)
        end
        extra_b[1, 1, i, 1] = layer.b[1, 1, extra_mapping[i], 1]
        if layer.bn_params != nothing
            extra_bn_scales[i] = layer.bn_params[extra_mapping[i]]
            extra_bn_biases[i] = layer.bn_params[old_channel_count+extra_mapping[i]]
        end
    end

    extra_w = atype()(extra_w)
    extra_next_w = next_layer != nothing ? atype()(extra_next_w) : nothing
    extra_b = atype()(extra_b)

    # Update the weights
    old_size = size(layer.w)
    w_old_2d = reshape(layer.w, :, size(layer.w)[end])
    w_extra_2d = reshape(extra_w, :, size(extra_w)[end])
    w_new_2d = hcat(w_old_2d, w_extra_2d)
    layer.w = Param(reshape(w_new_2d, old_size[1], old_size[2], old_size[3], :))

    layer.b = Param(reshape(vcat(reshape(layer.b, (size(layer.b, 3),)), reshape(extra_b, (extra_count,))), (1, 1, new_channel_count, 1)))

    if next_layer != nothing
        old_size = size(next_layer.w)
        w_old_2d = reshape(next_layer.w, :, size(next_layer.w)[end])
        w_extra_2d = reshape(extra_next_w, :, size(extra_next_w)[end])
        w_new_2d = vcat(w_old_2d, w_extra_2d)

        next_layer.w = Param(reshape(w_new_2d, old_size[1], old_size[2], :, old_size[4]))
    end
    if layer.bn_params != nothing
        layer.bn_params = Param(cat(layer.bn_params[1:old_channel_count], extra_bn_scales, layer.bn_params[old_channel_count+1:end], extra_bn_biases, dims=1))
    end

    return extra_mapping
end

function wider_inceptionA(inc::InceptionA, next::InceptionA, grow_ratio, add_noise::Bool=true)
    # Grow inside layers normally
    wider_conv(inc.c1_before_3, inc.c3, Int64(round(size(inc.c1_before_3.w, 4)*(grow_ratio))), add_noise)
    wider_conv(inc.c1_before_d3, inc.cd3_1, Int64(round(size(inc.c1_before_d3.w, 4)*(grow_ratio))), add_noise)
    wider_conv(inc.cd3_1, inc.cd3_2, Int64(round(size(inc.cd3_1.w, 4)*(grow_ratio))), add_noise)

    old_c1_count = size(inc.c1_alone.w, 4)
    old_c3_count = size(inc.c3.w, 4)
    old_cd3_count = size(inc.cd3_2.w, 4)
    old_c1_ap_count = size(inc.c1_after_pool.w, 4)

    new_c1_count = Int64(round(old_c1_count*(grow_ratio)))
    new_c3_count = Int64(round(old_c3_count*(grow_ratio)))
    new_cd3_count = Int64(round(old_cd3_count*(grow_ratio)))
    new_c1_ap_count = Int64(round(old_c1_ap_count*(grow_ratio)))

    c1_extra_mappings = wider_conv(inc.c1_alone, nothing, new_c1_count, add_noise)
    c3_extra_mappings = wider_conv(inc.c3, nothing, new_c3_count, add_noise)
    cd3_extra_mappings = wider_conv(inc.cd3_2, nothing, new_cd3_count, add_noise)
    c1_ap_extra_mappings = wider_conv(inc.c1_after_pool, nothing, new_c1_ap_count, add_noise)

    c1_mappings = vcat(1:old_c1_count, c1_extra_mappings)
    c3_mappings = vcat(1:old_c3_count, c3_extra_mappings) .+ old_c1_count
    cd3_mappings = vcat(1:old_cd3_count, cd3_extra_mappings) .+ (old_c1_count + old_c3_count)
    c1_ap_mappings = vcat(1:old_c1_ap_count, c1_ap_extra_mappings) .+ (old_c1_count + old_c3_count + old_cd3_count)
    total_mappings = vcat(c1_mappings, c3_mappings, cd3_mappings, c1_ap_mappings)

    old_channel_count = old_c1_count + old_c3_count + old_cd3_count + old_c1_ap_count
    new_channel_count = new_c1_count + new_c3_count + new_cd3_count + new_c1_ap_count

    copy_counts = zeros(old_channel_count)
    for i in 1:new_channel_count
        copy_counts[total_mappings[i]] += 1
    end

    for c in [next.c1_alone, next.c1_before_3, next.c1_before_d3, next.c1_after_pool]
        new_w = Array{Float32}(undef, (size(c.w)[1:2]..., new_channel_count, size(c.w)[4]))

        # Divide weights that are copied more than once
        c.w ./= atype()(reshape(copy_counts, (1,1,:,1)))

        array_w = convert(Array{Float32}, c.w)
        # Set the extra weights
        for i in 1:new_channel_count
            new_w[:, :, i, :] = array_w[:, :, total_mappings[i], :] .+ (add_noise ? randn()*0.01 : 0)
        end
        c.w = Param(atype()(new_w))
    end
end

function wider_inceptionA(inc::InceptionA, next::InceptionB, after_b::ConvBN, grow_ratio, add_noise::Bool=true)
    wider_conv(inc.c1_before_3, inc.c3, Int64(round(size(inc.c1_before_3.w, 4)*(grow_ratio))), add_noise)
    wider_conv(inc.c1_before_d3, inc.cd3_1, Int64(round(size(inc.c1_before_d3.w, 4)*(grow_ratio))), add_noise)
    wider_conv(inc.cd3_1, inc.cd3_2, Int64(round(size(inc.cd3_1.w, 4)*(grow_ratio))), add_noise)

    old_c1_count = size(inc.c1_alone.w, 4)
    old_c3_count = size(inc.c3.w, 4)
    old_cd3_count = size(inc.cd3_2.w, 4)
    old_c1_ap_count = size(inc.c1_after_pool.w, 4)

    new_c1_count = Int64(round(old_c1_count*(grow_ratio)))
    new_c3_count = Int64(round(old_c3_count*(grow_ratio)))
    new_cd3_count = Int64(round(old_cd3_count*(grow_ratio)))
    new_c1_ap_count = Int64(round(old_c1_ap_count*(grow_ratio)))

    c1_extra_mappings = wider_conv(inc.c1_alone, nothing, new_c1_count, add_noise)
    c3_extra_mappings = wider_conv(inc.c3, nothing, new_c3_count, add_noise)
    cd3_extra_mappings = wider_conv(inc.cd3_2, nothing, new_cd3_count, add_noise)
    c1_ap_extra_mappings = wider_conv(inc.c1_after_pool, nothing, new_c1_ap_count, add_noise)

    c1_mappings = vcat(1:old_c1_count, c1_extra_mappings)
    c3_mappings = vcat(1:old_c3_count, c3_extra_mappings) .+ old_c1_count
    cd3_mappings = vcat(1:old_cd3_count, cd3_extra_mappings) .+ (old_c1_count + old_c3_count)
    c1_ap_mappings = vcat(1:old_c1_ap_count, c1_ap_extra_mappings) .+ (old_c1_count + old_c3_count + old_cd3_count)
    total_mappings = vcat(c1_mappings, c3_mappings, cd3_mappings, c1_ap_mappings)

    old_channel_count = old_c1_count + old_c3_count + old_cd3_count + old_c1_ap_count
    new_channel_count = new_c1_count + new_c3_count + new_cd3_count + new_c1_ap_count

    copy_counts = zeros(old_channel_count)
    for i in 1:new_channel_count
        copy_counts[total_mappings[i]] += 1
    end

    for c in [next.c1_before_3, next.c1_before_d3]
        new_w = Array{Float32}(undef, (size(c.w)[1:2]..., new_channel_count, size(c.w)[4]))

        # Divide weights that are copied more than once
        c.w ./= atype()(reshape(copy_counts, (1,1,:,1)))

        array_w = convert(Array{Float32}, c.w)
        # Set the extra weights
        for i in 1:new_channel_count
            new_w[:, :, i, :] = array_w[:, :, total_mappings[i], :] .+ (add_noise ? randn()*0.01 : 0)
        end
        c.w = Param(atype()(new_w))
    end

    # Finally, need to change the layer after InceptionB as well due to the pass-through layer
    array_w = convert(Array{Float32}, after_b.w)
    w_first = array_w[:, :, 1:end-old_channel_count, :]
    w_last = array_w[:, :, end-old_channel_count+1:end, :]
    new_w_last = Array{Float32}(undef, (size(array_w)[1:2]..., new_channel_count, size(array_w)[4]))

    # Divide weights that are copied more than once
    w_last ./= (reshape(copy_counts, (1,1,:,1)))

    # Set the extra weights
    for i in 1:new_channel_count
        new_w_last[:, :, i, :] = w_last[:, :, total_mappings[i], :]
    end
    after_b.w = Param(atype()(cat(w_first, new_w_last, dims=3)))
end

function wider_inceptionB(inc::InceptionB, next::ConvBN, grow_ratio, add_noise::Bool=true)
    # Grow inside layers normally
    wider_conv(inc.c1_before_3, inc.c3, Int64(round(size(inc.c1_before_3.w, 4)*(grow_ratio))), add_noise)
    wider_conv(inc.c1_before_d3, inc.cd3_1, Int64(round(size(inc.c1_before_d3.w, 4)*(grow_ratio))), add_noise)
    wider_conv(inc.cd3_1, inc.cd3_2, Int64(round(size(inc.cd3_1.w, 4)*(grow_ratio))), add_noise)

    old_c3_count = size(inc.c3.w, 4)
    old_cd3_count = size(inc.cd3_2.w, 4)

    new_c3_count = Int64(round(old_c3_count*(grow_ratio)))
    new_cd3_count = Int64(round(old_cd3_count*(grow_ratio)))

    c3_extra_mappings = wider_conv(inc.c3, nothing, new_c3_count, add_noise)
    cd3_extra_mappings = wider_conv(inc.cd3_2, nothing, new_cd3_count, add_noise)

    c3_mappings = vcat(1:old_c3_count, c3_extra_mappings)
    cd3_mappings = vcat(1:old_cd3_count, cd3_extra_mappings) .+ old_c3_count
    total_mappings = vcat(c3_mappings, cd3_mappings)

    old_channel_count = old_c3_count + old_cd3_count
    new_channel_count = new_c3_count + new_cd3_count

    copy_counts = zeros(old_channel_count)
    for i in 1:new_channel_count
        copy_counts[total_mappings[i]] += 1
    end

    # Changing the layer after InceptionB
    array_w = convert(Array{Float32}, next.w)
    w_first = array_w[:, :, 1:old_channel_count, :]
    w_last = array_w[:, :, old_channel_count+1:end, :]
    new_w_first = Array{Float32}(undef, (size(array_w)[1:2]..., new_channel_count, size(array_w)[4]))

    # Divide weights that are copied more than once
    w_first ./= (reshape(copy_counts, (1,1,:,1)))

    # Set the extra weights
    for i in 1:new_channel_count
        new_w_first[:, :, i, :] = w_first[:, :, total_mappings[i], :] .+ (add_noise ? randn()*0.01 : 0)
    end
    next.w = Param(atype()(cat(new_w_first, w_last, dims=3)))
end

function test_wider_inception()
    (xtrn, ytrn), (xtst, ytst) = load_data()

    dtrn = minibatch(xtrn, ytrn, 50, xtype=atype())
    dtst = minibatch(xtst, ytst, 50, xtype=atype())

    model = create_inception_bn_sm_model(3, 10)

    results, model = train_results(dtrn, dtst, "inception_smaller.jld2", model, 50, false, false)
    wider = deepcopy(model)

    chosen_layer = 3
    growth_ratio = 1.25

    old_l = model.layers[chosen_layer]
    old_n = model.layers[chosen_layer+1]

    new_l = wider.layers[chosen_layer]
    new_n = wider.layers[chosen_layer+1]

    # instead of checking old model's layers manually, record results beforehand
    y_oldss = []
    for (x, y) in dtst
        push!(y_oldss, model(x))
    end

    wider_inceptionA(new_l, new_n, growth_ratio, false)

    @assert (size(new_l.c1_alone.w, 4) == Int64(round(size(old_l.c1_alone.w, 4) * (1 + growth_ratio))) &&
             size(new_l.c1_alone.b, 3) == Int64(round(size(old_l.c1_alone.b, 3) * (1 + growth_ratio))) &&
             size(new_l.c3.w, 4) == Int64(round(size(old_l.c3.w, 4) * (1 + growth_ratio))) &&
             size(new_l.c3.b, 3) == Int64(round(size(old_l.c3.b, 3) * (1 + growth_ratio))) &&
             size(new_l.cd3_2.w, 4) == Int64(round(size(old_l.cd3_2.w, 4) * (1 + growth_ratio))) &&
             size(new_l.cd3_2.b, 3) == Int64(round(size(old_l.cd3_2.b, 3) * (1 + growth_ratio))) &&
             size(new_l.c1_after_pool.w, 4) == Int64(round(size(old_l.c1_after_pool.w, 4) * (1 + growth_ratio))) &&
             size(new_l.c1_after_pool.b, 3) == Int64(round(size(old_l.c1_after_pool.b, 3) * (1 + growth_ratio))) &&

             size(new_n.c1_alone.w, 3) == Int64(round(size(old_n.c1_alone.w, 3) * (1 + growth_ratio))) &&
             size(new_n.c1_before_3.w, 3) == Int64(round(size(old_n.c1_before_3.w, 3) * (1 + growth_ratio))) &&
             size(new_n.c1_before_d3.w, 3) == Int64(round(size(old_n.c1_before_d3.w, 3) * (1 + growth_ratio))) &&
             size(new_n.c1_after_pool.w, 3) == Int64(round(size(old_n.c1_after_pool.w, 3) * (1 + growth_ratio)))
    ) "New model is not widened correctly"

    # Widening the next inceptionA
    wider_inceptionA(wider.layers[chosen_layer+1], wider.layers[chosen_layer+2], wider.layers[chosen_layer+4], growth_ratio, false)
    # Widening the next inceptionB
    wider_inceptionB(wider.layers[chosen_layer+2], wider.layers[chosen_layer+4], growth_ratio, false)

    sames = 0
    total = 0
    for (i, (x, y)) in enumerate(dtst)
        y_olds = y_oldss[i]
        y_news = wider(x)

        sames += sum(y_olds .- y_news .< 0.01)
        total += length(y_olds)
    end

    @assert sames/total == 1 "Function is not preserved"
    println("wider inceptionA test passed")
end

function test_wider_conv()
    (xtrn, ytrn), (xtst, ytst) = load_data()

    dtrn = minibatch(xtrn, ytrn, 50, xtype=atype())
    dtst = minibatch(xtst, ytst, 50, xtype=atype())

    narrow = 64
    wide = 72
    chosen_layer = 4

    cnn_model = create_cnn_model(3, 10, true)
    cnn_results, cnn_model = train_results(dtrn, dtst, "cnn.jld2", cnn_model, 5, false)
    cnn_wider = deepcopy(cnn_model)
    wider_conv(cnn_wider.layers[chosen_layer], cnn_wider.layers[chosen_layer+1], wide, false)

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
    mlp_wider = wider_mlp(mlp_model, chosen_layer, wide, false)

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

function test_random_pad_inception()
    (xtrn, ytrn), (xtst, ytst) = load_data()

    dtrn = minibatch(xtrn, ytrn, 50, xtype=atype())
    dtst = minibatch(xtst, ytst, 50, xtype=atype())

    model = create_inception_bn_sm_model(3, 10)

    results, model = train_results(dtrn, dtst, "inception_smaller.jld2", model, 50, false, false)
    wider = deepcopy(model)

    chosen_layer = 3
    growth_ratio = 1.25

    old_l = model.layers[chosen_layer]
    old_n = model.layers[chosen_layer+1]

    new_l = wider.layers[chosen_layer]
    new_n = wider.layers[chosen_layer+1]

    random_pad_inceptionA(new_l, new_n, growth_ratio)

    @assert (size(new_l.c1_alone.w, 4) == Int64(round(size(old_l.c1_alone.w, 4) * (1 + growth_ratio))) &&
             size(new_l.c1_alone.b, 3) == Int64(round(size(old_l.c1_alone.b, 3) * (1 + growth_ratio))) &&
             size(new_l.c3.w, 4) == Int64(round(size(old_l.c3.w, 4) * (1 + growth_ratio))) &&
             size(new_l.c3.b, 3) == Int64(round(size(old_l.c3.b, 3) * (1 + growth_ratio))) &&
             size(new_l.cd3_2.w, 4) == Int64(round(size(old_l.cd3_2.w, 4) * (1 + growth_ratio))) &&
             size(new_l.cd3_2.b, 3) == Int64(round(size(old_l.cd3_2.b, 3) * (1 + growth_ratio))) &&
             size(new_l.c1_after_pool.w, 4) == Int64(round(size(old_l.c1_after_pool.w, 4) * (1 + growth_ratio))) &&
             size(new_l.c1_after_pool.b, 3) == Int64(round(size(old_l.c1_after_pool.b, 3) * (1 + growth_ratio))) &&

             size(new_n.c1_alone.w, 3) == Int64(round(size(old_n.c1_alone.w, 3) * (1 + growth_ratio))) &&
             size(new_n.c1_before_3.w, 3) == Int64(round(size(old_n.c1_before_3.w, 3) * (1 + growth_ratio))) &&
             size(new_n.c1_before_d3.w, 3) == Int64(round(size(old_n.c1_before_d3.w, 3) * (1 + growth_ratio))) &&
             size(new_n.c1_after_pool.w, 3) == Int64(round(size(old_n.c1_after_pool.w, 3) * (1 + growth_ratio)))
    ) "New model is not widened correctly"

    random_pad_inceptionA(wider.layers[chosen_layer+1], wider.layers[chosen_layer+2], wider.layers[chosen_layer+4], growth_ratio)
    random_pad_inceptionB(wider.layers[chosen_layer+2], wider.layers[chosen_layer+4], growth_ratio)
    wider(dtst)

    println("random pad inceptionA test passed")
end

function random_pad_conv(layer, next_layer, new_channel_count)
    old_channel_count = size(layer.w, 4)
    if old_channel_count >= new_channel_count
        println("New channel count must be greater than old channel count")
        return nothing
    end

    # Initialize number of extra channels, extra arrays, random mapping, number of copies
    extra_count = new_channel_count - old_channel_count

    extra_w = xavier(size(layer.w)[1:3]..., extra_count)
    extra_next_w = next_layer != nothing ? xavier(size(next_layer.w)[1:2]..., extra_count, size(next_layer.w)[4]) : nothing
    extra_b = zeros(1, 1, extra_count, 1)

    if layer.bn_params != nothing
        # Initialize bn scales to 1 and bn biases to 0
        extra_bn_scales = atype()(ones(extra_count,))
        extra_bn_biases = atype()(zeros(extra_count,))
    end

    extra_w = atype()(extra_w)
    extra_next_w = next_layer != nothing ? atype()(extra_next_w) : nothing
    extra_b = atype()(extra_b)

    # Update the weights
    old_size = size(layer.w)
    w_old_2d = reshape(layer.w, :, size(layer.w)[end])
    w_extra_2d = reshape(extra_w, :, size(extra_w)[end])
    w_new_2d = hcat(w_old_2d, w_extra_2d)
    layer.w = Param(reshape(w_new_2d, old_size[1], old_size[2], old_size[3], :))

    layer.b = Param(reshape(vcat(reshape(layer.b, (size(layer.b, 3),)), reshape(extra_b, (extra_count,))), (1, 1, new_channel_count, 1)))

    if next_layer != nothing
        old_size = size(next_layer.w)
        w_old_2d = reshape(next_layer.w, :, size(next_layer.w)[end])
        w_extra_2d = reshape(extra_next_w, :, size(extra_next_w)[end])
        w_new_2d = vcat(w_old_2d, w_extra_2d)

        next_layer.w = Param(reshape(w_new_2d, old_size[1], old_size[2], :, old_size[4]))
    end
    if layer.bn_params != nothing
        layer.bn_params = Param(cat(layer.bn_params[1:old_channel_count], extra_bn_scales, layer.bn_params[old_channel_count+1:end], extra_bn_biases, dims=1))
    end
end

function random_pad_inceptionA(inc::InceptionA, next::InceptionA, grow_ratio)
    # Grow inside layers normally
    random_pad_conv(inc.c1_before_3, inc.c3, Int64(round(size(inc.c1_before_3.w, 4)*(grow_ratio))))
    random_pad_conv(inc.c1_before_d3, inc.cd3_1, Int64(round(size(inc.c1_before_d3.w, 4)*(grow_ratio))))
    random_pad_conv(inc.cd3_1, inc.cd3_2, Int64(round(size(inc.cd3_1.w, 4)*(grow_ratio))))

    old_c1_count = size(inc.c1_alone.w, 4)
    old_c3_count = size(inc.c3.w, 4)
    old_cd3_count = size(inc.cd3_2.w, 4)
    old_c1_ap_count = size(inc.c1_after_pool.w, 4)

    new_c1_count = Int64(round(old_c1_count*(grow_ratio)))
    new_c3_count = Int64(round(old_c3_count*(grow_ratio)))
    new_cd3_count = Int64(round(old_cd3_count*(grow_ratio)))
    new_c1_ap_count = Int64(round(old_c1_ap_count*(grow_ratio)))

    random_pad_conv(inc.c1_alone, nothing, new_c1_count)
    random_pad_conv(inc.c3, nothing, new_c3_count)
    random_pad_conv(inc.cd3_2, nothing, new_cd3_count)
    random_pad_conv(inc.c1_after_pool, nothing, new_c1_ap_count)

    old_channel_count = old_c1_count + old_c3_count + old_cd3_count + old_c1_ap_count
    new_channel_count = new_c1_count + new_c3_count + new_cd3_count + new_c1_ap_count

    for c in [next.c1_alone, next.c1_before_3, next.c1_before_d3, next.c1_after_pool]
        new_w = Array{Float32}(undef, (size(c.w)[1:2]..., new_channel_count, size(c.w)[4]))

        array_w = convert(Array{Float32}, c.w)
        new_w[:, :, 1:old_c1_count, :] = array_w[:, :, 1:old_c1_count, :]
        new_w[:, :, old_c1_count+1:new_c1_count, :] = xavier(size(array_w)[1:2]..., new_c1_count-old_c1_count, size(array_w)[4])

        new_w[:, :, new_c1_count+1:new_c1_count+old_c3_count, :] = array_w[:, :, old_c1_count+1:old_c1_count+old_c3_count, :]
        new_w[:, :, new_c1_count+old_c3_count+1:new_c1_count+new_c3_count, :] = xavier(size(array_w)[1:2]..., new_c3_count-old_c3_count, size(array_w)[4])

        new_w[:, :, new_c1_count+new_c3_count+1:new_c1_count+new_c3_count+old_cd3_count, :] = array_w[:, :, old_c1_count+old_c3_count+1:old_c1_count+old_c3_count+old_cd3_count, :]
        new_w[:, :, new_c1_count+new_c3_count+old_cd3_count+1:new_c1_count+new_c3_count+new_cd3_count, :] = xavier(size(array_w)[1:2]..., new_cd3_count-old_cd3_count, size(array_w)[4])

        new_w[:, :, new_c1_count+new_c3_count+new_cd3_count+1:new_c1_count+new_c3_count+new_cd3_count+old_c1_ap_count, :] = array_w[:, :, old_c1_count+old_c3_count+old_cd3_count+1:end, :]
        new_w[:, :, new_c1_count+new_c3_count+new_cd3_count+old_c1_ap_count+1:end, :] = xavier(size(array_w)[1:2]..., new_c1_ap_count-old_c1_ap_count, size(array_w)[4])

        c.w = Param(atype()(new_w))
    end
end

function random_pad_inceptionA(inc::InceptionA, next::InceptionB, after_b::ConvBN, grow_ratio)
    random_pad_conv(inc.c1_before_3, inc.c3, Int64(round(size(inc.c1_before_3.w, 4)*(grow_ratio))))
    random_pad_conv(inc.c1_before_d3, inc.cd3_1, Int64(round(size(inc.c1_before_d3.w, 4)*(grow_ratio))))
    random_pad_conv(inc.cd3_1, inc.cd3_2, Int64(round(size(inc.cd3_1.w, 4)*(grow_ratio))))

    old_c1_count = size(inc.c1_alone.w, 4)
    old_c3_count = size(inc.c3.w, 4)
    old_cd3_count = size(inc.cd3_2.w, 4)
    old_c1_ap_count = size(inc.c1_after_pool.w, 4)

    new_c1_count = Int64(round(old_c1_count*(grow_ratio)))
    new_c3_count = Int64(round(old_c3_count*(grow_ratio)))
    new_cd3_count = Int64(round(old_cd3_count*(grow_ratio)))
    new_c1_ap_count = Int64(round(old_c1_ap_count*(grow_ratio)))

    random_pad_conv(inc.c1_alone, nothing, new_c1_count)
    random_pad_conv(inc.c3, nothing, new_c3_count)
    random_pad_conv(inc.cd3_2, nothing, new_cd3_count)
    random_pad_conv(inc.c1_after_pool, nothing, new_c1_ap_count)

    old_channel_count = old_c1_count + old_c3_count + old_cd3_count + old_c1_ap_count
    new_channel_count = new_c1_count + new_c3_count + new_cd3_count + new_c1_ap_count

    for c in [next.c1_before_3, next.c1_before_d3]
        new_w = Array{Float32}(undef, (size(c.w)[1:2]..., new_channel_count, size(c.w)[4]))

        array_w = convert(Array{Float32}, c.w)
        new_w[:, :, 1:old_c1_count, :] = array_w[:, :, 1:old_c1_count, :]
        new_w[:, :, old_c1_count+1:new_c1_count, :] = xavier(size(array_w)[1:2]..., new_c1_count-old_c1_count, size(array_w)[4])

        new_w[:, :, new_c1_count+1:new_c1_count+old_c3_count, :] = array_w[:, :, old_c1_count+1:old_c1_count+old_c3_count, :]
        new_w[:, :, new_c1_count+old_c3_count+1:new_c1_count+new_c3_count, :] = xavier(size(array_w)[1:2]..., new_c3_count-old_c3_count, size(array_w)[4])

        new_w[:, :, new_c1_count+new_c3_count+1:new_c1_count+new_c3_count+old_cd3_count, :] = array_w[:, :, old_c1_count+old_c3_count+1:old_c1_count+old_c3_count+old_cd3_count, :]
        new_w[:, :, new_c1_count+new_c3_count+old_cd3_count+1:new_c1_count+new_c3_count+new_cd3_count, :] = xavier(size(array_w)[1:2]..., new_cd3_count-old_cd3_count, size(array_w)[4])

        new_w[:, :, new_c1_count+new_c3_count+new_cd3_count+1:new_c1_count+new_c3_count+new_cd3_count+old_c1_ap_count, :] = array_w[:, :, old_c1_count+old_c3_count+old_cd3_count+1:end, :]
        new_w[:, :, new_c1_count+new_c3_count+new_cd3_count+old_c1_ap_count+1:end, :] = xavier(size(array_w)[1:2]..., new_c1_ap_count-old_c1_ap_count, size(array_w)[4])

        c.w = Param(atype()(new_w))
    end

    # Finally, need to change the layer after InceptionB as well due to the pass-through layer
    array_w = convert(Array{Float32}, after_b.w)
    w_first = array_w[:, :, 1:end-old_channel_count, :]
    w_last = array_w[:, :, end-old_channel_count+1:end, :]
    new_w_last = Array{Float32}(undef, (size(array_w)[1:2]..., new_channel_count, size(array_w)[4]))

    # Set the extra weights
    new_w_last[:, :, 1:old_channel_count, :] = w_last
    new_w_last[:, :, old_channel_count+1:end, :] = xavier(size(w_last)[1:2]..., new_channel_count-old_channel_count, size(w_last)[4])

    after_b.w = Param(atype()(cat(w_first, new_w_last, dims=3)))
end
function random_pad_inceptionB(inc::InceptionB, next::ConvBN, grow_ratio)
    random_pad_conv(inc.c1_before_3, inc.c3, Int64(round(size(inc.c1_before_3.w, 4)*(grow_ratio))))
    random_pad_conv(inc.c1_before_d3, inc.cd3_1, Int64(round(size(inc.c1_before_d3.w, 4)*(grow_ratio))))
    random_pad_conv(inc.cd3_1, inc.cd3_2, Int64(round(size(inc.cd3_1.w, 4)*(grow_ratio))))

    old_c3_count = size(inc.c3.w, 4)
    old_cd3_count = size(inc.cd3_2.w, 4)

    new_c3_count = Int64(round(old_c3_count*(grow_ratio)))
    new_cd3_count = Int64(round(old_cd3_count*(grow_ratio)))

    random_pad_conv(inc.c3, nothing, new_c3_count)
    random_pad_conv(inc.cd3_2, nothing, new_cd3_count)

    old_channel_count = old_c3_count + old_cd3_count
    new_channel_count = new_c3_count + new_cd3_count

    array_w = convert(Array{Float32}, next.w)
    w_first = array_w[:, :, 1:end-old_channel_count, :]
    w_last = array_w[:, :, end-old_channel_count+1:end, :]
    new_w_last = Array{Float32}(undef, (size(array_w)[1:2]..., new_channel_count, size(array_w)[4]))

    # Set the extra weights
    new_w_last[:, :, 1:old_channel_count, :] = w_last
    new_w_last[:, :, old_channel_count+1:end, :] = xavier(size(w_last)[1:2]..., new_channel_count-old_channel_count, size(w_last)[4])

    next.w = Param(atype()(cat(w_first, new_w_last, dims=3)))
end
