


function get_discriminator(args)
    ndim = args["ndim"]
    return Chain(Conv((4, 4), 3 => ndim, stride=(2, 2), pad=1, bias=false),
                 BatchNorm(ndim, elu),
                 # 2. block
                 Conv((4, 4), ndim => 2 * ndim, stride=(2, 2), pad=1, bias=false),
                 BatchNorm(2 * ndim, elu),
                 # 3. block
                 Conv((4, 4), 2 * ndim => 4 * ndim, stride=(2, 2), pad=1, bias=false),
                 BatchNorm(4 * ndim, elu),
                 # 4. block
                 Conv((4, 4), 4 * ndim => 4 * ndim, bias=false),
                 BatchNorm(4 * ndim, elu),
                 x -> flatten(x),
                 Dense(4 * ndim, 10),
                 x -> softmax(x))

end


function get_generator(args)
    ndim = args["ndim"]
    ...
end
