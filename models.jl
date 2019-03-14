
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

# Convolutional layer
mutable struct Conv; w; b; f; p; end
(c::Conv)(x) = c.f.(pool(conv4(c.w, dropout(x,c.p)) .+ c.b))
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu;pdrop=0) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f, pdrop)

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
