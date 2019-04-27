@inline _reddims(y) = ((i for i=1:ndims(y)-2)..., ndims(y))

"""
Initializes an identity convolutional filter of the given size.
"""
function id_filter(w1, w2, cx, cy)
    w = zeros(w1, w2, cx, cy)
    center_1::Int = ceil(w1/2)
    center_2::Int = ceil(w2/2)
    for i in 1:cy
        tmp = zeros(w1, w2, cy)
        tmp[center_1, center_2, i] = 1
        w[:, :, :, i] = tmp
    end
    return w
end

"""
Net2WiderDeeperNet method for Convs.
Creates a new convolutional layer with identity filters that fit with the previous layer.
Function is preserved during the deepening.
If layer uses batchnorm, training minibatches are needed for initialization.
"""
function deeper_conv(layers, layer_index, dtrn=nothing)
    prev_layer = layers[layer_index]
    if !(prev_layer isa Conv) && !(prev_layer isa ConvBN)
        print("Layer is not convolutional!")
    end

    prev_w = prev_layer.w
    deeper_w = param(size(prev_w, 1), size(prev_w, 2), size(prev_w, 4), size(prev_w, 4); init=id_filter, atype=atype())
    deeper_b = param0(1, 1, size(prev_w, 4), 1; atype=atype())
    if prev_layer isa Conv
        deeper_layer = Conv(deeper_w, deeper_b, prev_layer.f, Int(floor(size(prev_w, 1)/2)), prev_layer.stride)
    else
        deeper_layer = ConvBN(deeper_w, deeper_b, prev_layer.f, Int(floor(size(prev_w, 1)/2)), prev_layer.stride, Param(convert(atype(), bnparams(size(prev_w, 4)))), bnmoments())
        # Forward computation to set the new bn moments and params correctly
        for (x, y) in dtrn
            h = x
            for i in 1:layer_index
                h = layers[i](h)
            end
            ah = convert(Array, h)
            dims = _reddims(ah)
            _lazy_init!(deeper_layer.bn_moments, ah)

            mu = mean(ah, dims=dims)
            # sigma2 = var(h; corrected=false, mean=mu, dims=dims)
            sigma2 = Statistics._var(ah, false, convert(Array, mu), dims)

            _update_moments!(deeper_layer.bn_moments, mu, sigma2)
        end

        deeper_layer.bn_moments.mean = convert(atype(), deeper_layer.bn_moments.mean)
        deeper_layer.bn_moments.var = convert(atype(), deeper_layer.bn_moments.var)
        eps = 1e-5
        gamma = sqrt.(deeper_layer.bn_moments.var .+ eps)
        beta = deeper_layer.bn_moments.mean

        deeper_layer.bn_params = vcat(reshape(gamma, size(gamma, 3)), reshape(beta, size(beta, 3)))
    end
    insert!(layers, layer_index+1, deeper_layer)
end

"""
Net2WiderDeeperNet method for Inception modules.
Creates new convolutional layers to deepen the module.
Function is preserved during the deepening.
"""
function deeper_inception(layers, layer_index, dtrn=nothing)
    inc_layer = layers[layer_index]
    if !(inc_layer isa InceptionA) && !(inc_layer isa InceptionB)
        print("Layer is not an Inception module!")
    end

    if inc_layer isa InceptionA
        
    else

        # Forward computation to set the new bn moments and params correctly
        for (x, y) in dtrn
            h = x
            for i in 1:layer_index
                h = layers[i](h)
            end
            ah = convert(Array, h)
            dims = _reddims(ah)
            _lazy_init!(deeper_layer.bn_moments, ah)

            mu = mean(ah, dims=dims)
            # sigma2 = var(h; corrected=false, mean=mu, dims=dims)
            sigma2 = Statistics._var(ah, false, convert(Array, mu), dims)

            _update_moments!(deeper_layer.bn_moments, mu, sigma2)
        end

        deeper_layer.bn_moments.mean = convert(atype(), deeper_layer.bn_moments.mean)
        deeper_layer.bn_moments.var = convert(atype(), deeper_layer.bn_moments.var)
        eps = 1e-5
        gamma = sqrt.(deeper_layer.bn_moments.var .+ eps)
        beta = deeper_layer.bn_moments.mean

        deeper_layer.bn_params = vcat(reshape(gamma, size(gamma, 3)), reshape(beta, size(beta, 3)))
    end
    insert!(layers, layer_index+1, deeper_layer)
end


function test_deeper_conv(with_bn=true)
    (xtrn, ytrn), (xtst, ytst) = load_data()

    dtrn = minibatch(xtrn, ytrn, 50, xtype=atype())
    dtst = minibatch(xtst, ytst, 50, xtype=atype())

    chosen_layer = 4

    cnn_model = create_cnn_model(3, 10, with_bn)
    file_name = with_bn ? "cnn.jld2" : "cnn_no_bn.jld2"
    cnn_results, cnn_model = train_results(dtrn, dtst, file_name, cnn_model, 5, false)
    old_layer_count = length(cnn_model.layers)

    cnn_deeper = deepcopy(cnn_model)
    deeper_conv(cnn_deeper.layers, chosen_layer, dtrn)

    @assert (length(cnn_model.layers) == old_layer_count) "Old model is modified"
    @assert (length(cnn_deeper.layers) == old_layer_count + 1) "New model is not deepened correctly"

    sames = 0
    total = 0
    for (x, y) in dtst
        y_olds = x
        y_news = x

        for i in 1:chosen_layer
            if sum(y_news .- y_olds) != 0
                println("Old layers are changed! i = ", i)
            end
            y_olds = cnn_model.layers[i](y_olds)
            y_news = cnn_deeper.layers[i](y_news)
        end

        y_news = cnn_deeper.layers[chosen_layer+1](y_news)
        differents = abs.(y_olds .- y_news) .< 0.01

        sames_mb = Int(sum(differents))

        sames += sames_mb
        total += length(y_olds)
    end

    @show sames/total
    @assert sames/total == 1 "Function is not preserved"
    println("deeper conv test passed")
end
