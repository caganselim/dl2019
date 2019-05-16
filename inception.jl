# Inception modules
struct InceptionA
    c1_alone::ConvBN
    c1_before_3::ConvBN
    c1_before_d3::ConvBN
    c1_after_pool::ConvBN

    c3::ConvBN
    cd3_1::ConvBN
    cd3_2::ConvBN
    pool_mode::Int
end

function InceptionA(cx, num_1, num_1_before_3, num_3, num_1_before_d3, num_d3, num_1_after_pool, pool_mode)
    c1_alone = ConvBN(1, 1, cx, num_1)
    c1_before_3 = ConvBN(1, 1, cx, num_1_before_3)
    c1_before_d3 = ConvBN(1, 1, cx, num_1_before_d3)
    c1_after_pool = ConvBN(1, 1, cx, num_1_after_pool)

    c3 = ConvBN(3, 3, num_1_before_3, num_3, padding=1)
    cd3_1 = ConvBN(3, 3, num_1_before_d3, num_d3, padding=1)
    cd3_2 = ConvBN(3, 3, num_d3, num_d3, padding=1)
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
    c1_before_3::ConvBN
    c1_before_d3::ConvBN

    c3::ConvBN
    cd3_1::ConvBN
    cd3_2::ConvBN
end
function InceptionB(cx, num_1_before_3, num_3, num_1_before_d3, num_d3)
    c1_before_3 = ConvBN(1, 1, cx, num_1_before_3)
    c1_before_d3 = ConvBN(1, 1, cx, num_1_before_d3)

    c3 = ConvBN(3, 3, num_1_before_3, num_3, padding=1, stride=2)
    cd3_1 = ConvBN(3, 3, num_1_before_d3, num_d3, padding=1)
    cd3_2 = ConvBN(3, 3, num_d3, num_d3, padding=1, stride=2)
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


#------------------------ Net2DeeperNet versions -------------------------------

# Deeper version of InceptionA
struct InceptionADeeper
    c1_before_3::ConvBN
    c1_before_d3::ConvBN
    c1_after_pool::ConvBN
    c1_alone::ConvBN

    c3s::Array{ConvBN}
    cd3s::Array{ConvBN}

    pool_mode::Int
end

# This constructor is used during random initialization (baseline for Net2DeeperNet)
function InceptionADeeper(cx, num_1, num_1_before_3, num_3, num_1_before_d3, num_d3, num_1_after_pool, pool_mode, deepening_factor::Int=2)
    c1_before_3 = ConvBN(1, 1, cx, num_1_before_3)
    c1_before_d3 = ConvBN(1, 1, cx, num_1_before_d3)
    c1_after_pool = ConvBN(1, 1, cx, num_1_after_pool)
    c1_alone = ConvBN(1, 1, cx, num_1)

    c3s = [ConvBN(3, 3, num_1_before_3, num_3, padding=1)]
    cd3s = [ConvBN(3, 3, num_1_before_d3, num_d3, padding=1), ConvBN(3, 3, num_d3, num_d3, padding=1)]

    for i in 1:deepening_factor
        push!(c3s, ConvBN(3, 3, num_3, num_3, padding=1))
        push!(cd3s, ConvBN(3, 3, num_d3, num_d3, padding=1), ConvBN(3, 3, num_d3, num_d3, padding=1))
    end

    return InceptionADeeper(c1_before_3, c1_before_d3, c1_after_pool, c1_alone, c3s, cd3s, pool_mode)
end

function (i::InceptionADeeper)(x)
    y1 = i.c1_alone(x)
    y2 = i.c1_before_3(x); for l in i.c3s; y2 = l(y2); end;
    y3 = i.c1_before_d3(x); for l in i.cd3s; y3 = l(y3); end;
    y4 = i.c1_after_pool(pool(x, window=3, stride=1, padding=1, mode=i.pool_mode))
    old_size = size(y1)

    y1 = reshape(y1, :, size(y1)[end])
    y2 = reshape(y2, :, size(y2)[end])
    y3 = reshape(y3, :, size(y3)[end])
    y4 = reshape(y4, :, size(y4)[end])

    y_2d = vcat(y1, y2, y3, y4)
    return reshape(y_2d, old_size[1], old_size[2], :, old_size[4])
end

# Deeper version of InceptionB
struct InceptionBDeeper
    c1_before_3::ConvBN
    c1_before_d3::ConvBN

    c3s::Array{ConvBN}
    cd3s::Array{ConvBN}
end

# This constructor is used during random initialization (baseline for Net2DeeperNet)
function InceptionBDeeper(cx, num_1_before_3, num_3, num_1_before_d3, num_d3, deepening_factor::Int=2)
    c1_before_3 = ConvBN(1, 1, cx, num_1_before_3)
    c1_before_d3 = ConvBN(1, 1, cx, num_1_before_d3)

    c3s = [ConvBN(3, 3, num_1_before_3, num_3, padding=1, stride=2)]
    cd3s = [ConvBN(3, 3, num_1_before_d3, num_d3, padding=1), ConvBN(3, 3, num_d3, num_d3, padding=1, stride=2)]

    for i in 1:deepening_factor
        push!(c3s, ConvBN(3, 3, num_3, num_3, padding=1))
        push!(cd3s, ConvBN(3, 3, num_d3, num_d3, padding=1), ConvBN(3, 3, num_d3, num_d3, padding=1))
    end
    return InceptionBDeeper(c1_before_3, c1_before_d3, c3s, cd3s)
end

function (i::InceptionBDeeper)(x)
    y1 = i.c1_before_3(x); for l in i.c3s; y1 = l(y1); end;
    y2 = i.c1_before_d3(x); for l in i.cd3s; y2 = l(y2); end;
    y3 = pool(x, window=3, stride=2, padding=1)
    old_size = size(y1)

    y1 = reshape(y1, :, size(y1)[end])
    y2 = reshape(y2, :, size(y2)[end])
    y3 = reshape(y3, :, size(y3)[end])

    y_2d = vcat(y1, y2, y3)
    return reshape(y_2d, old_size[1], old_size[2], :, old_size[4])
end


#------------------------------ Network creator functions ----------------------

"Builds an original Inception-BN network model"
function create_inception_bn_model(num_channels::Int, num_classes::Int)
    Chain(
        ConvBN(7, 7, num_channels, 64, padding=3, stride=2),
        Pool(3, 2, 0, 0),

        ConvBN(1, 1, 64, 64),
        ConvBN(3, 3, 64, 192, padding=1),
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

"""
Builds a smaller version of Inception-BN network model modified to take 32x32 images.
This is the version used in the experiments.
"""
function create_inception_bn_sm_model(num_channels::Int, num_classes::Int, deeper=false)
    IA = deeper ? InceptionADeeper : InceptionA
    IB = deeper ? InceptionBDeeper : InceptionB

    Chain(
        ConvBN(3, 3, num_channels, 48),
        ConvBN(3, 3, 48, 96),

        IA(96, 26, 26, 26, 26, 35, 13, 2),
        IA(100, 26, 26, 35, 26, 35, 26, 2),
        IB(122, 53, 70, 26, 35),

        Pool(5, 3, 0, 2),
        ConvBN(1, 1, 227, 64),
        Flatten(1024),
        Linear(1024, num_classes)
    )
end
