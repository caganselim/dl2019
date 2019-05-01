# A chain of layers used to build models
struct Chain
    layers::Array
    Chain(layers...) = new(collect(layers))
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)
(c::Chain)(d::Data) = mean(c(x,y) for (x,y) in d)

# Linear layer
mutable struct Linear; w; b; end
Linear(i::Int,o::Int) = Linear(param(o,i),param0(o))
(l::Linear)(x) = l.w * x .+ l.b

# Softmax classifier layer
mutable struct SoftmaxCls; w; b; end
SoftmaxCls(i::Int,o::Int) = SoftmaxCls(param(o,i),param0(o))
(l::SoftmaxCls)(x) = softmax(l.w * x .+ l.b)

# Dense (normal) layer
mutable struct Dense; w; b; f; pdrop; end
Dense(i::Int,o::Int,f=relu; pdrop=0) = Dense(param(o,i),param0(o),f,pdrop)
(l::Dense)(x) = l.f.(l.w * dropout(x,l.pdrop) .+ l.b)

# Flatten layer. I had to make it a struct because otherwise it does not save to file
struct Flatten; n::Int end
(r::Flatten)(x) = reshape(x, (r.n,:))

# Convolutional + pooling layer
mutable struct ConvPool; w; b; f; p; end
(c::ConvPool)(x) = c.f.(pool(conv4(c.w, dropout(x,c.p)) .+ c.b))
ConvPool(w1::Int,w2::Int,cx::Int,cy::Int,f=relu;pdrop=0) = ConvPool(param(w1,w2,cx,cy), param0(1,1,cy,1), f, pdrop)

# Convolutional layer
mutable struct Conv; w; b; f; padding; stride; end
function (c::Conv)(x)
    c.f.(conv4(c.w, x, padding=c.padding, stride=c.stride) .+ c.b)
end
function Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu;padding=0,stride=1)
    return Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f, padding, stride)
end

function my_bn(x, bn_moments, bn_params)
    eps = 1e-5
    ivar = 1 ./ sqrt.(bn_moments.var .+ eps)

    g = reshape(bn_params[1:size(x, 3)], (1, 1, size(x, 3), 1))
    b = reshape(bn_params[size(x, 3)+1:end], (1, 1, size(x, 3), 1))
    return g .* (x .- bn_moments.mean) .* ivar .+ b
end

# Convolutional + Batchnorm layer
mutable struct ConvBN; w; b; f; padding; stride; bn_params; bn_moments; end
function (c::ConvBN)(x)
    y_conv = conv4(c.w, x, padding=c.padding, stride=c.stride) .+ c.b
    c.f.(batchnorm(y_conv, c.bn_moments, c.bn_params))
end
function (c::ConvBN)(x, skip_bn::Bool)
    y_conv = conv4(c.w, x, padding=c.padding, stride=c.stride) .+ c.b
    if skip_bn
        return c.f.(y_conv)
    end
    return c.f.(batchnorm(y_conv, c.bn_moments, c.bn_params))
end
function ConvBN(w1::Int,w2::Int,cx::Int,cy::Int,f=relu;padding=0,stride=1)
    return ConvBN(param(w1,w2,cx,cy), param0(1,1,cy,1), f, padding, stride, Param(convert(atype(), bnparams(cy))), bnmoments())
end

# Pooling layer
struct Pool
    window
    stride
    padding
    mode
    Pool(window=2, stride=2, padding=0, mode=0) = new(window, stride, padding, mode)
end
(p::Pool)(x) = pool(x, window=p.window, stride=p.stride, padding=p.padding, mode=p.mode)

include("inception.jl")

"Builds a CNN model consisting of 3x3 filters and 2x2 max pools"
function create_cnn_model(num_channels::Int, num_classes::Int, use_bn::Bool=false)
    C = use_bn ? ConvBN : Conv
    Chain(
        C(3, 3, num_channels, 32),
        C(3, 3, 32, 48),
        Pool(), # 14x14

        C(3, 3, 48, 64),
        C(3, 3, 64, 96),
        Pool(), # 5x5

        C(3, 3, 96, 128),
        C(3, 3, 128, 192),

        Flatten(192),
        Linear(192, num_classes)
    )
end

"Builds an MLP model with any number of hidden layers of any given unit sizes"
function create_mlp_model(i::Int, o::Int, h_units::Int ... ; f=relu)
    weight_dims = [i]
    for u in h_units
        push!(weight_dims, u, u)
    end
    push!(weight_dims, o)

    # Unit counts are used as such: [(i, h1), (h1, h2), (h2, h3), ..., (hn, o)]
    layer_params = []
    for i::Int in 1:(length(weight_dims)/2)
        d1 = weight_dims[2*i - 1]
        d2 = weight_dims[2*i]
        push!(layer_params, [d1, d2, f])
    end

    # Set the last function to be identity
    layer_params[end][3] = identity

    @show layer_params

    layers = (Dense(lp...) for lp in layer_params)
    return Chain(layers...)
end

# Utility function that calculates the output channel size of Inception modules
function out_channel_count(i::InceptionA)
    return size(i.c1_alone.w, 4) + size(i.c3.w, 4) + size(i.cd3_2.w, 4) + size(i.c1_after_pool.w, 4)
end
function out_channel_count(i::InceptionB)
    return size(i.c3.w, 4) + size(i.cd3_2.w, 4) + size(i.c1_before_3.w, 3)
end
