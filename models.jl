# A chain of layers used to build models
struct Chain
    layers
    Chain(layers...) = new(layers)
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
struct Conv; w; b; f; padding; stride; bn_params; bn_moments; end
function (c::Conv)(x)
    c.f.(conv4(c.w, batchnorm(x, c.bn_moments, c.bn_params), padding=c.padding, stride=c.stride) .+ c.b)
end
function Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu;padding=0,stride=1)
    return Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f, padding, stride, convert(atype(), bnparams(cx)), bnmoments())
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

# Inception modules
struct InceptionA
    c1_alone::Conv
    c1_before_3::Conv
    c1_before_d3::Conv
    c1_after_pool::Conv

    c3::Conv
    cd3_1::Conv
    cd3_2::Conv
    pool_mode::Int
end

function InceptionA(cx, num_1, num_1_before_3, num_3, num_1_before_d3, num_d3, num_1_after_pool, pool_mode)
    c1_alone = Conv(1, 1, cx, num_1)
    c1_before_3 = Conv(1, 1, cx, num_1_before_3)
    c1_before_d3 = Conv(1, 1, cx, num_1_before_d3)
    c1_after_pool = Conv(1, 1, cx, num_1_after_pool)

    c3 = Conv(3, 3, num_1_before_3, num_3, padding=1)
    cd3_1 = Conv(3, 3, num_1_before_d3, num_d3, padding=1)
    cd3_2 = Conv(3, 3, num_d3, num_d3, padding=1)
    return InceptionA(c1_alone, c1_before_3, c1_before_d3, c1_after_pool, c3, cd3_1, cd3_2, pool_mode)
end

function (i::InceptionA)(x)
    y1 = i.c1_alone(x)
    y2 = i.c3(i.c1_before_3(x))
    y3 = i.cd3_2(i.cd3_1(i.c1_before_d3(x)))
    y4 = i.c1_after_pool(pool(x, window=3, stride=1, padding=1, mode=i.pool_mode))
    old_size = size(y1)

    y1 = reshape(y1, :, size(y1)[end])
    y2 = reshape(y2, :, size(y2)[end])
    y3 = reshape(y3, :, size(y3)[end])
    y4 = reshape(y4, :, size(y4)[end])

    y_2d = vcat(y1, y2, y3, y4)
    return reshape(y_2d, old_size[1], old_size[2], :, old_size[4])
end

struct InceptionB
    c1_before_3::Conv
    c1_before_d3::Conv

    c3::Conv
    cd3_1::Conv
    cd3_2::Conv
end
function InceptionB(cx, num_1_before_3, num_3, num_1_before_d3, num_d3)
    c1_before_3 = Conv(1, 1, cx, num_1_before_3)
    c1_before_d3 = Conv(1, 1, cx, num_1_before_d3)

    c3 = Conv(3, 3, num_1_before_3, num_3, padding=1, stride=2)
    cd3_1 = Conv(3, 3, num_1_before_d3, num_d3, padding=1)
    cd3_2 = Conv(3, 3, num_d3, num_d3, padding=1, stride=2)
    return InceptionB(c1_before_3, c1_before_d3, c3, cd3_1, cd3_2)
end

function (i::InceptionB)(x)
    y1 = i.c3(i.c1_before_3(x))
    y2 = i.cd3_2(i.cd3_1(i.c1_before_d3(x)))
    y3 = pool(x, window=3, stride=2, padding=1)
    old_size = size(y1)

    y1 = reshape(y1, :, size(y1)[end])
    y2 = reshape(y2, :, size(y2)[end])
    y3 = reshape(y3, :, size(y3)[end])

    y_2d = vcat(y1, y2, y3)
    return reshape(y_2d, old_size[1], old_size[2], :, old_size[4])
end

"Builds an Inception-BN network model"
function create_inception_bn_model(num_channels::Int, num_classes::Int)
    Chain(
        Conv(7, 7, num_channels, 64, padding=3, stride=2),
        Pool(3, 2, 0, 0),

        Conv(1, 1, 64, 64),
        Conv(3, 3, 64, 192, padding=1),
        Pool(3, 2, 0, 0),

        InceptionA(192, 64, 64, 64, 64, 96, 32, 2),
        InceptionA(256, 64, 64, 96, 64, 96, 64, 2),
        InceptionB(320, 128, 160, 64, 96),

        InceptionA(576, 224, 64, 96, 96, 128, 128, 2),
        InceptionA(576, 192, 96, 128, 96, 128, 128, 2),
        InceptionA(576, 160, 128, 160, 128, 160, 128, 2),
        InceptionA(608, 96, 128, 192, 160, 192, 128, 2),
        InceptionB(608, 128, 192, 192, 256),

        InceptionA(1056, 352, 192, 320, 160, 224, 128, 2),
        InceptionA(1024, 352, 192, 320, 192, 224, 128, 0),
        Pool(7, 1, 0, 2),
        Dense(1024, num_classes)
    )
end

"Builds an Inception-BN network model modified to take 32x32 images"
function create_inception_bn_cifar_model(num_channels::Int, num_classes::Int)
    Chain(
        Conv(3, 3, num_channels, 64),

        Conv(1, 1, 64, 64),
        Conv(3, 3, 64, 192),

        InceptionA(192, 64, 64, 64, 64, 96, 32, 2),
        InceptionA(256, 64, 64, 96, 64, 96, 64, 2),
        InceptionB(320, 128, 160, 64, 96),

        InceptionA(576, 224, 64, 96, 96, 128, 128, 2),
        InceptionA(576, 192, 96, 128, 96, 128, 128, 2),
        InceptionA(576, 160, 128, 160, 128, 160, 128, 2),
        InceptionA(608, 96, 128, 192, 160, 192, 128, 2),
        InceptionB(576, 128, 192, 192, 256),

        InceptionA(1056, 352, 192, 320, 160, 224, 128, 2),
        InceptionA(1024, 352, 192, 320, 192, 224, 128, 0),
        Pool(7, 1, 0, 2),
        Flatten(1024),
        Linear(1024, num_classes)
    )
end
function create_inception_bn_smaller_model(num_channels::Int, num_classes::Int)
    Chain(
        Conv(3, 3, num_channels, 48),
        Conv(3, 3, 48, 128),

        InceptionA(128, 48, 48, 48, 48, 64, 24, 2),
        InceptionA(184, 48, 48, 64, 48, 64, 48, 2),
        InceptionB(224, 96, 128, 48, 64), # 14x14x576

        Pool(5, 3, 0, 2), # 4x4x576
        Conv(1, 1, 416, 64),
        Flatten(1024),
        # Dense(1024, 1024, pdrop=0.7),
        Linear(1024, num_classes)
    )
end

"Builds a CNN model consisting of 3x3 filters and 2x2 max pools"
function create_cnn_model(num_channels::Int, num_classes::Int)
    Chain(
        Conv(3, 3, num_channels, 32),
        Conv(3, 3, 32, 48),
        Pool(), # 14x14

        Conv(3, 3, 48, 64),
        Conv(3, 3, 64, 96),
        Pool(), # 5x5

        Conv(3, 3, 96, 128),
        Conv(3, 3, 128, 192),

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
