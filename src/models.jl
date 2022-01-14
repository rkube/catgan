

export get_discriminator, get_discriminator_v2, get_generator, get_generator_v2

function get_discriminator(args)
    mdim = args["mdim"]
    # Cifar images start out as (32)x(32)x(x)
    return Chain(Conv((4, 4), 3 => mdim, stride=(2, 2), pad=1, bias=false),
                 # Image is now (16)x(16)x(8)
                 BatchNorm(mdim, elu),
                 # 2. block
                 Conv((4, 4), mdim => 2 * mdim, stride=(2, 2), pad=1, bias=false),
                 # Image is now (8)x(8)x(16)
                 BatchNorm(2 * mdim, elu),
                 # 3. block
                 Conv((4, 4), 2 * mdim => 4 * mdim, stride=(2, 2), pad=1, bias=false),
                 # Image is now (4)x(4)x(32)
                 BatchNorm(4 * mdim, elu),
                 # 4. block
                 Conv((4, 4), 4 * mdim => 4 * mdim, bias=false),
                 # Image is now (1)x(1)x(32)
                 BatchNorm(4 * mdim, elu),
                 x -> flatten(x),
                 # state is now (32)x(batch_size)
                 Dense(4 * mdim, 10),
                 x -> softmax(x))

end


function get_discriminator_v2(args)
    D2 = Chain(Conv((3, 3), 3 => 96, elu),
               Conv((3, 3), 96=>96, elu),
               Conv((3, 3), 96=>96, elu),
               MaxPool((2, 2), stride=2),
               BatchNorm(96, elu),
               Conv((3, 3), 96=>192, elu),
               Conv((3, 3), 192=>192, elu),
               Conv((3, 3), 192=>192, elu),
               MaxPool((3, 3), stride=2),
               BatchNorm(192, elu),
               Conv((3, 3), 192=>192, elu),
               Conv((1, 1), 192=>192, elu),
               Conv((1, 1), 192=>10),
               GlobalMeanPool(), x -> elu.(x),
               flatten, softmax)
end



function get_generator(args)
    mdim = args["mdim"] # Model dimension
    noise_dim = args["noise_dim"]   # Latent dimension

    # The input to the generator is a 1x1 image with noise_dim channels
    return Chain(ConvTranspose((4, 4), noise_dim => 4 * mdim, bias=false),
                 # Now imega is (4)x(4)x(4*mdim)
                 BatchNorm(4 * mdim, elu),
                 ConvTranspose((4, 4), 4 * mdim => 2 * mdim, stride=(2, 2), pad=1, bias=false),
                 # Now image is (8)x(8)x(2*mdim)
                 BatchNorm(2 * mdim, elu),
                 ConvTranspose((4, 4), 2 * mdim => mdim, stride=(2, 2), pad=1, bias=false),
                 # Now image is (16)x(16)x(mdim)
                 BatchNorm(mdim, elu),
                 ConvTranspose((4, 4), mdim => 3, tanh, stride=(2, 2), pad=1, bias=false))
                 # Now image is (32)x(32)x(3)
end


function get_generator_v2(args)
    noise_dim = args["noise_dim"]

    return Chain(flatten, Dense(args["noise_dim"], 8*8*192, elu),
                 x -> reshape(x, (8, 8, 192, 100)),
                 ConvTranspose((2, 2), 192=>192, elu, stride=(2, 2)),
                 Conv((5, 5), 192 => 96, elu, pad=SamePad()),
                 Conv((5, 5), 96 => 96, elu, pad=SamePad()),
                 ConvTranspose((2, 2), 96 => 96, stride=(2, 2)),
                 Conv((5, 5), 96 => 96, elu, pad=SamePad()),
                 Conv((5, 5), 96 => 3, elu, pad=SamePad()));

end
