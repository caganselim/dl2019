
struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)
(c::Chain)(d::Data) = mean(c(x,y) for (x,y) in d)

# Dense (normal) layer
struct Dense; w; b; f; pdrop; end
Dense(i::Int,o::Int,f=relu; pdrop=0) = Dense(param(o,i),param0(o),f,pdrop)
(l::Dense)(x) = l.f.(l.w * dropout(x,l.pdrop) .+ l.b)

# Convolutional layer
struct Conv; w; b; f; p; end
(c::Conv)(x) = c.f.(pool(conv4(c.w, dropout(x,c.p)) .+ c.b))
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu;pdrop=0) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f, pdrop)
