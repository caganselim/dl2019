
struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)
(c::Chain)(d::Data) = mean(c(x,y) for (x,y) in d)

# Dense (normal) layer
mutable struct Dense; w; b; f; pdrop; end
Dense(i::Int,o::Int,f=relu; pdrop=0) = Dense(param(o,i),param0(o),f,pdrop)
(l::Dense)(x) = l.f.(l.w * dropout(x,l.pdrop) .+ l.b)

# Convolutional + pooling layer
mutable struct ConvPool; w; b; f; p; end
(c::ConvPool)(x) = c.f.(pool(conv4(c.w, dropout(x,c.p)) .+ c.b))
ConvPool(w1::Int,w2::Int,cx::Int,cy::Int,f=relu;pdrop=0) = ConvPool(param(w1,w2,cx,cy), param0(1,1,cy,1), f, pdrop)

# Convolutional layer
struct Conv; w; b; f; padding; stride; end
function (c::Conv)(x)
    c.f.(conv4(c.w, x, padding=c.padding, stride=c.stride) .+ c.b) #todo: add BN
end
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu;padding=0,stride=1) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f, padding, stride)

# Pooling layer
struct Pool
    window
    stride
    padding
    mode
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
struct InceptionB
    c1_before_3::Conv
    c1_before_d3::Conv

    c3::Conv
    cd3_1::Conv
    cd3_2::Conv
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
    return KnetArray(cat(Array(y1), Array(y2), Array(y3), Array(y4), dims=3))
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
    y2 = i.c3(i.c1_before_3(x))
    y3 = i.cd3_2(i.cd3_1(i.c1_before_d3(x)))
    y4 = pool(x, window=3, stride=2, padding=1)
    return KnetArray(cat(Array(y2), Array(y3), Array(y4), dims=3))
end

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

function create_inception_bn_small_model(num_channels::Int, num_classes::Int)
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
        InceptionB(608, 128, 192, 192, 256),

        InceptionA(1056, 352, 192, 320, 160, 224, 128, 2),
        InceptionA(1024, 352, 192, 320, 192, 224, 128, 0),
        Pool(7, 1, 0, 2),
        x -> reshape(x, (1024,:)),
        Dense(1024, num_classes)
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
